"""Offline session storage, recording, deterministic analysis, and reports.

Model loading is deliberately local-only. A missing model produces an explicit
stage error; it never silently turns an unchecked result into a judgement.
"""
from __future__ import annotations

import copy
import csv
import gc
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SESSIONS = DATA / "sessions"
EXPORTS = ROOT / "exports"
RATE = 16000
os.environ.setdefault("HF_HOME", str(DATA / "model-cache"))

CATEGORIES = ["R_L", "TH", "V_B", "VOWEL", "EPENTHESIS", "F_H"]
DEFAULTS = {
    "audio": {"sample_rate": 16000, "denoise": False, "device": "システム既定"},
    "vad": {"min_silence_sec": 0.6},
    "asr": {"model": "small.en", "compute_type": "int8"},
    "phoneme": {"model": "facebook/wav2vec2-lv-60-espeak-cv-ft", "min_confidence": 0.6},
    "intent": {"lm": "distilgpt2", "min_delta_logprob": 3.0, "lexicon_size": 30000, "min_zipf": 3.0},
    "local_mode": {"practice": "post_analysis", "ollama_model": "llama3.2:3b",
                   "max_reply_words": 40, "turn_silence_sec": 1.2},
    "tutor": {"asr_model": "base.en", "min_confidence": 0.75, "strength": "focus",
              "max_retries": 2, "no_repeat_turns": 3, "pron_categories": list(CATEGORIES)},
    "tts": {"piper_exe": "piper", "piper_voice": ""},
}
DEFAULT_MODEL_IDS = {"phoneme": DEFAULTS["phoneme"]["model"], "intent": DEFAULTS["intent"]["lm"]}

# One lock for loading shared models: the tutor warm-up thread and a turn must
# not load the same large model twice.
_load_lock = threading.RLock()


def config():
    with (ROOT / "config.toml").open("rb") as f:
        loaded = tomllib.load(f)
    merged = copy.deepcopy(DEFAULTS)
    for section, values in loaded.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
        else:
            merged[section] = values
    return merged


def save_config(c):
    # TOML accepts JSON strings, numbers, booleans and arrays of strings.
    lines = []
    for section, values in c.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    (ROOT / "config.toml").write_text("\n".join(lines), encoding="utf-8")


def model_dir(kind, model_id):
    """The default models keep the existing folders; others get their own."""
    if model_id == DEFAULT_MODEL_IDS[kind]:
        return DATA / "models" / kind
    return DATA / "models" / f"{kind}-{re.sub(r'[^A-Za-z0-9._-]', '_', model_id)}"


def new_session(mode, scenario=None):
    SESSIONS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = SESSIONS / f"{stamp}_{mode}"
    n = 1
    while path.exists():
        path = SESSIONS / f"{stamp}_{mode}_{n}"
        n += 1
    path.mkdir()
    return path


def read_wav(path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != RATE:
        from scipy.signal import resample_poly
        divisor = math.gcd(sr, RATE)
        y = resample_poly(y, RATE // divisor, sr // divisor).astype("float32")
    return np.asarray(y, dtype="float32")


def write_wav(path, y):
    import soundfile as sf
    sf.write(str(path), np.asarray(y, dtype="float32"), RATE, subtype="PCM_16")


class Recorder:
    """The sounddevice callback only copies samples; analysis cannot stall capture.

    Audio is kept in memory and written when the session stops. If the app is
    closed mid-session, that recording is intentionally discarded.
    """

    def __init__(self):
        self.stream = None
        self.blocks = []
        self.lock = threading.Lock()
        self.started = None
        self.last_turn_end = 0
        self.last_level = 0.0

    def start(self, device=None):
        import sounddevice as sd
        if self.stream is not None:
            raise RuntimeError("録音はすでに開始しています")
        self.blocks = []
        self.last_turn_end = 0
        self.started = time.monotonic()

        def callback(indata, frames, timing, status):
            block = indata[:, 0].copy()
            with self.lock:
                self.blocks.append(block)
                self.last_level = float(np.sqrt(np.mean(block * block)))

        self.stream = sd.InputStream(samplerate=RATE, channels=1,
                                     dtype="float32", blocksize=1024,
                                     device=device, callback=callback)
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise

    def snapshot(self):
        with self.lock:
            blocks = list(self.blocks)
        return np.concatenate(blocks) if blocks else np.empty(0, dtype="float32")

    def seconds(self):
        with self.lock:
            return sum(len(b) for b in self.blocks) / RATE

    def stop(self, path):
        if self.stream is None:
            raise RuntimeError("録音していません")
        self.stream.stop()
        self.stream.close()
        self.stream = None
        y = self.snapshot()
        write_wav(path, y)
        return y

    def status(self):
        if not self.stream:
            return "停止中"
        elapsed = time.monotonic() - self.started
        level = min(100, self.last_level * 400)
        meter = "▮" * round(level / 10) + "▯" * (10 - round(level / 10))
        return f"録音中 {int(elapsed // 60)}:{int(elapsed % 60):02d} · 入力レベル {meter} {level:.0f}%"


# ---------------------------------------------------------------- stage 1

def denoise(y):
    """Optional DeepFilterNet pass. Raises when DeepFilterNet is not installed."""
    import torch
    from df.enhance import enhance, init_df
    from scipy.signal import resample_poly
    global _df
    with _load_lock:
        if _df is None:
            _df = init_df()
    model, state, _ = _df
    sr = state.sr()
    divisor = math.gcd(sr, RATE)
    up = resample_poly(y, sr // divisor, RATE // divisor).astype("float32")
    out = enhance(model, state, torch.from_numpy(up).unsqueeze(0)).squeeze(0).numpy()
    down = resample_poly(out, RATE // divisor, sr // divisor).astype("float32")
    return down[:len(y)] if len(down) >= len(y) else np.pad(down, (0, len(y) - len(down)))


_df = None

# ---------------------------------------------------------------- stage 2

_vad_model = None


def speech_spans(y, silence=0.6):
    global _vad_model
    if len(y) == 0:
        return []
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    with _load_lock:
        if _vad_model is None:
            _vad_model = load_silero_vad()
        spans = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(y)), _vad_model,
                                      sampling_rate=RATE,
                                      min_silence_duration_ms=round(silence * 1000),
                                      return_seconds=True)
    return [{"start": float(s["start"]), "end": float(s["end"])} for s in spans]


# ---------------------------------------------------------------- stage 3

_asr = {}


def release_asr():
    _asr.clear()
    gc.collect()


def transcribe(y, model_name="small.en", compute_type="int8"):
    y = np.asarray(y, dtype="float32")
    if len(y) < RATE // 5 or float(np.sqrt(np.mean(y * y))) < 0.002:
        return []
    segments, _ = load_asr(model_name, compute_type).transcribe(
        y, language="en", word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False, no_speech_threshold=0.5,
        initial_prompt="Um, I think, uh, this is an example. Er, let me see.")
    result = []
    for seg in segments:
        words = seg.words or []
        if seg.end <= seg.start or not words or all((w.end or 0) <= (w.start or 0) for w in words):
            continue
        result.append({"start": float(seg.start), "end": float(seg.end),
                       "text": seg.text.strip(),
                       "words": [{"w": w.word.strip(), "start": float(w.start),
                                  "end": float(w.end)} for w in words]})
    return result


# ---------------------------------------------------------------- stage 4-5

_phoneme_model = None
_espeak = None
_espeak_lock = threading.Lock()


def release_phoneme():
    global _phoneme_model
    _phoneme_model = None
    gc.collect()


def _espeak_backend():
    global _espeak
    if _espeak is None:
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        _espeak = EspeakBackend("en-us", with_stress=False)
    return _espeak


def clean_word(w):
    return re.sub(r"[^a-z']", "", str(w).lower()).strip("'")


def phonemize_words(words):
    """espeak IPA per word. The result always has one list per input word;
    words with no letters (numbers, symbols) get an empty list."""
    from phonemizer.separator import Separator
    clean = [clean_word(w) for w in words]
    todo = [i for i, w in enumerate(clean) if w]
    result = [[] for _ in clean]
    if not todo:
        return result
    with _espeak_lock:
        out = _espeak_backend().phonemize([clean[i] for i in todo],
                                          separator=Separator(phone=" ", word="", syllable=""),
                                          strip=True, njobs=1)
    for i, p in zip(todo, out):
        result[i] = p.split()
    return result


expected_ipa = phonemize_words


def recognized_phones(y, model_id):
    global _phoneme_model
    with _load_lock:
        if _phoneme_model is None or _phoneme_model[0] != model_id:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
            local = model_dir("phoneme", model_id)
            if not (local / "vocab.json").exists():
                raise FileNotFoundError("音素モデルがありません。run.ps1 models を実行してください")
            vocab = json.loads((local / "vocab.json").read_text(encoding="utf-8"))
            if vocab and all(isinstance(v, dict) for v in vocab.values()):
                vocab = next(iter(vocab.values()))
            try:
                feature = Wav2Vec2FeatureExtractor.from_pretrained(str(local), local_files_only=True)
            except Exception:
                feature = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=RATE,
                                                   padding_value=0, do_normalize=True,
                                                   return_attention_mask=True)
            model = Wav2Vec2ForCTC.from_pretrained(str(local), local_files_only=True)
            model.eval()
            _phoneme_model = (model_id, feature, model, vocab)
    _, feature, model, vocab = _phoneme_model
    import torch
    batch = feature(y, sampling_rate=RATE, return_tensors="pt")
    with torch.inference_mode():
        prob = torch.softmax(model(batch.input_values).logits[0].float(), dim=-1).numpy()
    inv = {i: p for p, i in vocab.items()}
    blank = model.config.pad_token_id if model.config.pad_token_id is not None else vocab.get("<pad>", 0)
    ids = prob.argmax(axis=1)
    runs = []
    start = 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            token = inv.get(int(ids[start]), "")
            if ids[start] != blank and token and not token.startswith("<") and token not in ("|", " "):
                runs.append({"phone": token, "start": start * len(y) / max(1, len(ids)) / RATE,
                             "end": i * len(y) / max(1, len(ids)) / RATE,
                             "conf": float(prob[start:i, ids[start]].mean())})
            start = i
    return runs


def phones_in_word(heard, start, end):
    """Each recognised phone belongs to exactly one word: the one containing
    its midpoint. Phones straddling a boundary are no longer counted twice."""
    return [p for p in heard if start <= (p["start"] + p["end"]) / 2 < end]


# ---------------------------------------------------------------- stage 6

def edit_alignment(expected, heard):
    n, m = len(expected), len(heard)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1,
                          d[i-1][j-1] + (expected[i-1] != heard[j-1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1] + (expected[i-1] != heard[j-1]):
            ops.append(("eq" if expected[i-1] == heard[j-1] else "sub", i-1, j-1)); i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j] + 1:
            ops.append(("del", i-1, None)); i -= 1
        else:
            ops.append(("ins", None, j-1)); j -= 1
    return list(reversed(ops))


VOWEL_CHARS = set("aeiouyæɑɒɔəɚɛɜɝɪʊʌɐᵻɨøœɯɤ")
EPENTHESIS_VOWELS = {"ə", "ʊ", "u", "uː", "ɯ", "o", "oʊ", "ɔ", "ɨ", "ᵻ", "ɪ", "i", "iː"}
VOWEL_SET = {"æ", "ʌ", "ɑ", "ɑː", "ɒ"}


def is_vowel(phone):
    return any(ch in VOWEL_CHARS for ch in phone)


def category(a, b, insertion=False, final=False):
    pair = {a, b}
    if insertion:
        return "EPENTHESIS" if final and b in EPENTHESIS_VOWELS else "OTHER"
    if pair <= {"ɹ", "r", "l"} and len(pair) == 2: return "R_L"
    if pair & {"θ", "ð"} and pair & {"s", "z", "d"}: return "TH"
    if pair == {"v", "b"}: return "V_B"
    if pair <= VOWEL_SET and len(pair) == 2 and not (pair == {"ɑ", "ɑː"}): return "VOWEL"
    if pair == {"f", "h"}: return "F_H"
    return "OTHER"


def compare_word(ipa, ph, threshold):
    """Deviations of one word. An insertion is epenthesis only when it comes
    after the last expected phone of a word that ends in a consonant."""
    errors = []
    ops = edit_alignment(ipa, [p["phone"] for p in ph])
    ends_in_consonant = bool(ipa) and not is_vowel(ipa[-1])
    for k, (op, i, j) in enumerate(ops):
        if op == "eq": continue
        actual = ph[j]["phone"] if j is not None else ""
        expected = ipa[i] if i is not None else ""
        conf = ph[j]["conf"] if j is not None else min((p["conf"] for p in ph), default=0)
        if conf < threshold: continue
        final = op == "ins" and ends_in_consonant and all(o[1] is None for o in ops[k + 1:])
        errors.append({"category": category(expected, actual, op == "ins", final),
                       "expected": expected, "heard": actual,
                       "conf": round(conf, 3), "operation": op})
    return errors


def pronunciation(utterances, y, c, threshold=None):
    threshold = threshold or c["phoneme"]["min_confidence"]
    errors = []
    for utt in utterances:
        words = utt.get("words", [])
        if not words: continue
        clip = y[round(utt["start"] * RATE):round(utt["end"] * RATE)]
        if len(clip) < RATE // 5: continue
        heard = recognized_phones(clip, c["phoneme"]["model"])
        ipas = phonemize_words([w["w"] for w in words])
        for word, ipa in zip(words, ipas):
            start = word["start"] - utt["start"]
            end = word["end"] - utt["start"]
            ph = phones_in_word(heard, start, end)
            word["expected"] = "".join(ipa)
            word["heard"] = "".join(p["phone"] for p in ph)
            word["conf"] = round(min((p["conf"] for p in ph), default=0), 3)
            # Alignment across independently estimated word/phone times is noisy.
            # Low-confidence phones, missing audio or unknown words stay unjudged.
            if not ipa or not ph or word["conf"] < threshold: continue
            for e in compare_word(ipa, ph, threshold):
                errors.append({"utt": utt["id"], "word": word["w"], "start": word["start"],
                               "end": word["end"], "expected_word": word["expected"],
                               "heard_word": word["heard"], **e})
    return errors


# ---------------------------------------------------------------- stage 7

SUBSTITUTIONS = [("ɹ", "l", "R_L"), ("θ", "s", "TH"), ("ð", "z", "TH"), ("ð", "d", "TH"),
                 ("v", "b", "V_B"), ("æ", "ʌ", "VOWEL"), ("æ", "ɑː", "VOWEL"),
                 ("ʌ", "ɑː", "VOWEL"), ("f", "h", "F_H")]
_lexicon = None


def lexicon(c=None):
    """Real English words indexed by their espeak IPA (same symbols as stage 5)."""
    global _lexicon
    c = c or config()
    size, min_zipf = int(c["intent"]["lexicon_size"]), float(c["intent"]["min_zipf"])
    with _load_lock:
        if _lexicon is None or _lexicon[0] != (size, min_zipf):
            from wordfreq import top_n_list, zipf_frequency
            words = [w for w in top_n_list("en", size)
                     if re.fullmatch(r"[a-z]+", w) and (len(w) > 1 or w in ("a", "i"))
                     and zipf_frequency(w, "en") >= min_zipf]
            index = {}
            for w, ipa in zip(words, phonemize_words(words)):
                if ipa: index.setdefault(" ".join(ipa), []).append(w)
            _lexicon = ((size, min_zipf), index)
    return _lexicon[1]


def one_substitution_words(word, c=None):
    """Words reachable from `word` by exactly one confusable substitution."""
    ipa = phonemize_words([word])[0]
    index = lexicon(c)
    source = clean_word(word)
    found = []
    for k, phone in enumerate(ipa):
        for a, b, cat in SUBSTITUTIONS:
            if phone not in (a, b): continue
            variant = ipa[:k] + [b if phone == a else a] + ipa[k + 1:]
            for w in index.get(" ".join(variant), []):
                if w != source and (w, cat) not in [(x[0], x[1]) for x in found]:
                    found.append((w, cat, "".join(ipa), "".join(variant)))
    return found


_lm = None


def release_lm():
    global _lm
    _lm = None
    gc.collect()


def sentence_logprob(sentence, model_id):
    global _lm
    with _load_lock:
        if _lm is None or _lm[0] != model_id:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            local = model_dir("intent", model_id)
            if not (local / "config.json").exists():
                raise FileNotFoundError("言い間違い判定モデルがありません。run.ps1 models を実行してください")
            tok = AutoTokenizer.from_pretrained(str(local), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(str(local), local_files_only=True)
            model.eval()
            _lm = model_id, tok, model
    import torch
    _, tok, model = _lm
    ids = tok((tok.bos_token or "") + sentence, return_tensors="pt").input_ids
    if ids.shape[1] < 2: return 0.0
    with torch.inference_mode():
        logits = model(ids).logits[:, :-1]
        scores = torch.log_softmax(logits.float(), dim=-1)
        return float(scores.gather(-1, ids[:, 1:].unsqueeze(-1)).sum())


def replace_nth_word(text, word, nth, replacement):
    matches = list(re.finditer(r"\b" + re.escape(word) + r"\b", text, flags=re.I))
    if nth >= len(matches): return text
    m = matches[nth]
    rep = replacement.capitalize() if m.group(0)[:1].isupper() else replacement
    return text[:m.start()] + rep + text[m.end():]


def intent_candidates(utterances, c, logprob=None):
    """Dictionary words one confusable substitution away, kept only when the
    language model finds the replaced sentence clearly more natural."""
    logprob = logprob or (lambda s: sentence_logprob(s, c["intent"]["lm"]))
    found = []
    for utt in utterances:
        original = utt["text"]
        base = None
        seen = Counter()
        for word in utt.get("words", []):
            source = clean_word(word["w"])
            if not source: continue
            nth = seen[source]; seen[source] += 1
            best = None
            for candidate, cat, src_ipa, cand_ipa in one_substitution_words(source, c):
                alternate = replace_nth_word(original, source, nth, candidate)
                if alternate == original: continue
                if base is None: base = logprob(original)
                delta = logprob(alternate) - base
                if best is None or delta > best["delta_logprob"]:
                    best = {"utt": utt["id"], "word": source, "intended": candidate,
                            "category": cat, "delta_logprob": round(delta, 2),
                            "original": original, "candidate": alternate,
                            "start": word.get("start", 0), "end": word.get("end", 0),
                            "expected": cand_ipa, "heard": src_ipa}
            if best and best["delta_logprob"] >= c["intent"]["min_delta_logprob"]:
                found.append(best)
    return found


# ---------------------------------------------------------------- stage 8

_grammar_tool = None
_grammar_lock = threading.Lock()
GRAMMAR_SKIP = ("PUNCTUATION", "CASING", "TYPOGRAPHY", "WHITESPACE", "UPPERCASE", "COMMA")


def grammar(utterances):
    global _grammar_tool
    import language_tool_python
    result = []
    with _grammar_lock:
        if _grammar_tool is None:
            _grammar_tool = language_tool_python.LanguageTool("en-US")
        for utt in utterances:
            text = utt["text"]
            for match in _grammar_tool.check(text):
                rule = str(getattr(match, "rule_id", None) or getattr(match, "ruleId", ""))
                category_name = str(getattr(match, "category", ""))
                if any(s in (rule + " " + category_name).upper() for s in GRAMMAR_SKIP): continue
                if not match.replacements: continue
                offset = int(match.offset)
                length = int(getattr(match, "error_length", None) or getattr(match, "errorLength", 0))
                suggestion = match.replacements[0]
                result.append({"utt": utt["id"], "text": text, "original": text[offset:offset + length],
                               "suggestion": suggestion,
                               "corrected": text[:offset] + suggestion + text[offset + length:],
                               "offset": offset, "length": length, "rule": rule,
                               "message": str(getattr(match, "message", ""))})
    return result


# ---------------------------------------------------------------- stage 9

FILLERS = {"uh", "um", "er", "erm", "ah", "uhm", "hmm"}


def fluency(utterances, spans, duration):
    words = [w for u in utterances for w in u.get("words", [])]
    count = len(words)
    speech = sum(max(0, s["end"] - s["start"]) for s in spans)
    pauses = []
    prev = 0.0
    for s in spans:
        if s["start"] - prev >= 0.25: pauses.append(s["start"] - prev)
        prev = s["end"]
    if duration - prev >= 0.25: pauses.append(duration - prev)
    runs = []
    for s in spans:
        runs.append(sum(s["start"] <= (w["start"] + w["end"]) / 2 <= s["end"] for w in words))
    fillers = sum(w["w"].strip(".,?!- ").lower() in FILLERS for w in words)
    return {"wpm": round(count * 60 / duration, 1) if duration else 0,
            "articulation_rate": round(count * 60 / speech, 1) if speech else 0,
            "pause_ratio": round(sum(pauses) / duration, 3) if duration else 0,
            "mean_run": round(sum(runs) / len(runs), 1) if runs else 0,
            "fillers": fillers, "word_count": count}


FLUENCY_LABELS = {"wpm": "1分あたりの語数", "articulation_rate": "発話速度（無音を除く・語/分）",
                  "pause_ratio": "沈黙の割合", "mean_run": "続けて話した語数の平均",
                  "fillers": "つなぎ言葉の回数"}


def top_three(errors, intents, grammar_issues):
    counts = Counter(("pron", e["category"]) for e in errors)
    counts.update(("intent", e["category"]) for e in intents)
    counts.update(("grammar", e["rule"]) for e in grammar_issues)
    rank = {"pron": 0, "intent": 1, "grammar": 2}
    other_utts = len({e.get("utt") for e in errors if e["category"] == "OTHER"})
    return [{"kind": k, "category": cat, "count": n}
            for (k, cat), n in sorted(counts.items(), key=lambda x: (-x[1], rank[x[0][0]], x[0][1]))
            if n >= 2 and (k != "pron" or cat != "OTHER" or (n >= 5 and other_utts >= 3))][:3]


# ---------------------------------------------------------------- pipeline

STAGES = ["前処理", "発話区間", "文字起こし", "音素認識・照合", "言い間違い候補", "文法",
          "流暢さ", "レポート", "完了"]


def analyze(path, c=None, progress=None, partner=""):
    c = c or config()
    progress = progress or (lambda stage, fraction=None: None)
    raw = read_wav(path / "audio.wav")
    duration = len(raw) / RATE
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    result = {"session": {"id": path.name, "mode": meta.get("mode", "separate"),
                          "practice": meta.get("practice"), "scenario": meta.get("scenario"),
                          "created": meta.get("created"), "duration_sec": round(duration, 2)},
              "utterances": [], "pron_errors": [], "intent_candidates": [],
              "grammar": [], "fluency": {}, "top3": [], "tutor_corrections": meta.get("tutor_corrections", []),
              "stage_errors": []}
    if partner.strip(): (path / "partner.txt").write_text(partner, encoding="utf-8")

    def stage(name, func, default):
        progress(name, STAGES.index(name) / (len(STAGES) - 1) if name in STAGES else None)
        try: return func()
        except Exception as e:
            result["stage_errors"].append({"stage": name, "error": f"{type(e).__name__}: {e}"})
            return default

    y = raw
    if c["audio"].get("denoise"):
        y = stage("前処理", lambda: denoise(raw), raw)
    else:
        progress("前処理", 0)
    spans = stage("発話区間", lambda: speech_spans(y, c["vad"]["min_silence_sec"]), [])

    def do_asr():
        out = []
        for s in spans:
            clip = y[round(s["start"]*RATE):round(s["end"]*RATE)]
            for seg in transcribe(clip, c["asr"]["model"], c["asr"]["compute_type"]):
                words = [{**w, "start": round(w["start"] + s["start"], 3),
                          "end": round(w["end"] + s["start"], 3)} for w in seg["words"]]
                out.append({"id": len(out), "start": round(seg["start"] + s["start"], 3),
                            "end": round(seg["end"] + s["start"], 3), "speaker": "user",
                            "text": seg["text"], "words": words})
        return out
    result["utterances"] = stage("文字起こし", do_asr, [])
    release_asr()
    if result["utterances"]:
        result["pron_errors"] = stage("音素認識・照合", lambda: pronunciation(result["utterances"], y, c), [])
        release_phoneme()
        result["intent_candidates"] = stage("言い間違い候補", lambda: intent_candidates(result["utterances"], c), [])
        release_lm()
        result["grammar"] = stage("文法", lambda: grammar(result["utterances"]), [])
    short_spans = stage("流暢さ", lambda: speech_spans(y, 0.25), spans)
    result["fluency"] = fluency(result["utterances"], short_spans, duration)
    if not result["utterances"]:
        result["stage_errors"].append({"stage": "分析", "error": "文字起こしがないため、発音・文法・言い間違いは判定できません"})
    result["top3"] = top_three(result["pron_errors"], result["intent_candidates"], result["grammar"])
    (path / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    progress("レポート", STAGES.index("レポート") / (len(STAGES) - 1))
    write_report(path, result, c)
    update_history(result)
    progress("完了", 1.0)
    return result


HISTORY_FIELDS = ["id", "mode", "wpm", "articulation_rate", "pause_ratio", "mean_run", "fillers"]


def update_history(result):
    DATA.mkdir(exist_ok=True)
    path = DATA / "history.csv"
    rows = read_history()
    flu = result["fluency"]
    row = {"id": result["session"]["id"], "mode": result["session"]["mode"],
           **{k: flu.get(k, "") for k in HISTORY_FIELDS[2:]}}
    rows = sorted([r for r in rows if r.get("id") != row["id"]] + [row], key=lambda r: r["id"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_history():
    path = DATA / "history.csv"
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def claude_text(r):
    top3 = "、".join(f"{t['kind']}:{t['category']}（{t['count']}件）" for t in r["top3"]) or "なし"
    intents = "; ".join(f"'{x['word']}'→'{x['intended']}'（{x['category']}, 差{x['delta_logprob']}）: {x['original']}"
                        for x in r["intent_candidates"]) or "なし"
    gram = "; ".join(f"{g['text']} → {g.get('corrected', g['suggestion'])}（{g['rule']}）" for g in r["grammar"]) or "なし"
    flu = ", ".join(f"{FLUENCY_LABELS.get(k, k)} {v}" for k, v in r["fluency"].items() if k in FLUENCY_LABELS)
    return ("あなたは英会話講師です。以下は私の英会話練習の文字起こしと、自動分析の結果です。\n"
            "あなたは音声を聞いていないので、発音については分析結果だけを根拠にしてください。\n\n"
            "1. 重点3つを、それぞれ短く解説してください\n"
            "2. 文脈から見て、言い間違いの可能性がある箇所を確認してください\n"
            "3. 重点のうち1つについて、英語で短いドリルをしてください\n\n"
            f"【重点】{top3}\n"
            f"【言い間違いの可能性】{intents}\n"
            f"【文法の指摘】{gram}\n"
            f"【流暢さ】{flu}\n"
            f"【文字起こし】{' '.join(u['text'] for u in r['utterances'])}")


PRACTICE = {
    "R_L": ["Turn on the light, right now.", "I read a long letter.", "The road is really long.", "Please look at the red light.", "I like rice for lunch."],
    "TH": ["I think this is the third one.", "Thank you for the thoughtful note.", "They are there together.", "The thin thread is strong.", "This is the thing I meant."],
    "V_B": ["The very big van is blue.", "I have a vivid memory.", "The boat moved very slowly.", "Visit the village by the river.", "Bring the blue vase."],
    "VOWEL": ["The cat cut the bag.", "I had a cup of coffee.", "That black hat is mine.", "The bus stopped at the park.", "Dad packed the lunch."],
    "EPENTHESIS": ["Put the desk back.", "I asked for help.", "Stop at the next street.", "I work late.", "Please check the list."],
    "F_H": ["He found fresh food.", "I have half a fish.", "The fan is very hot.", "Find the house first.", "Her friend feels fine."],
    "OTHER": ["Please say the word again.", "Could you repeat that slowly?", "Let me say it one more time.",
              "I will speak a little more slowly.", "Can you hear the difference?"],
}
EXPLAIN = {
    "R_L": "r と l の取り違え。l は舌先を上の歯の裏に付け、r は舌先をどこにも付けずに少し奥へ引きます。",
    "TH": "th が s・z・d になっています。舌先を上下の歯で軽くはさみ、息を通します。",
    "V_B": "v が b になっています。上の歯を下唇に軽く当て、声を出しながら息を通します。",
    "VOWEL": "æ（cat）・ʌ（cut）・ɑ（cot）の区別。æ は口を横に大きく、ʌ は短く弱く、ɑ は縦に大きく開けます。",
    "EPENTHESIS": "語末の子音のあとに余計な母音が入っています（desk → desku）。最後の子音で止めます。",
    "F_H": "f と h の混同。f は上の歯を下唇に当てて息を出します。",
    "OTHER": "上記以外の音のずれ。単語と音素の時刻合わせによる誤検出も含み得るので、音声で確認してください。",
}
KIND_LABEL = {"pron": "発音", "intent": "言い間違い", "grammar": "文法", "context": "文脈・言葉選び（参考）"}


def _tutor_mark(r, kind, category, word):
    for t in r.get("tutor_corrections", []):
        if t.get("kind") == kind and t.get("category") == category and \
                clean_word(t.get("original", "")) == clean_word(word):
            return "訂正済み・" + ("合格" if t.get("passed") else "不合格")
    return ""


def _highlight(text, marks):
    """marks: list of (start, end, css_class, title). Overlaps keep the first."""
    marks = sorted(marks)
    out, pos = [], 0
    for s, e, cls, title in marks:
        if s < pos or e <= s: continue
        out.append(html.escape(text[pos:s]))
        out.append(f"<mark class='{cls}' title='{html.escape(title, quote=True)}'>{html.escape(text[s:e])}</mark>")
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _utterance_marks(u, r):
    text, marks, cursor = u["text"], [], 0
    pron = {(e["start"], e["word"]): e["category"] for e in r["pron_errors"] if e["utt"] == u["id"]}
    intent = {(e.get("start"), e["word"]): e["intended"] for e in r["intent_candidates"] if e["utt"] == u["id"]}
    for w in u.get("words", []):
        token = w["w"].strip()
        if not token: continue
        at = text.find(token, cursor)
        if at < 0: continue
        cursor = at + len(token)
        cw = clean_word(token)
        if (w["start"], cw) in intent:
            marks.append((at, cursor, "m-intent", f"言い間違いの可能性 → {intent[(w['start'], cw)]}"))
        elif (w["start"], w["w"]) in pron:
            marks.append((at, cursor, "m-pron", f"発音 {pron[(w['start'], w['w'])]}"))
    for g in r["grammar"]:
        if g["utt"] == u["id"] and "offset" in g:
            marks.append((g["offset"], g["offset"] + max(1, g["length"]), "m-grammar", f"文法 → {g['suggestion']}"))
    return _highlight(text, marks)


def _session_datetime(sess):
    if sess.get("created"): return sess["created"]
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})(\d{2})(\d{2})?", sess["id"])
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}" if m else "—"


def write_report(path, r, c=None):
    esc = lambda x: html.escape(str(x), quote=True)
    sess = r["session"]
    audio = read_wav(path / "audio.wav") if (path / "audio.wav").exists() else np.empty(0)
    dur = len(audio) / RATE
    piper_ok = [True]

    def clip(name, start, end):
        start, end = max(0, start - .2), min(dur, end + .2)
        if end <= start: return ""
        write_wav(path / name, audio[round(start*RATE):round(end*RATE)])
        return f"<audio controls preload='none' src='{esc(name)}'></audio>"

    def reference(name, text):
        if not text or not piper_ok[0]: return ""
        try:
            piper(text, path / name, c)
            return f"<audio controls preload='none' src='{esc(name)}'></audio>"
        except Exception:
            piper_ok[0] = False
            return ""

    utt_by_id = {u["id"]: u for u in r["utterances"]}
    focus_parts = []
    for n, t in enumerate(r["top3"]):
        cat, kind = t["category"], t["kind"]
        pool = {"pron": r["pron_errors"], "intent": r["intent_candidates"], "grammar": r["grammar"]}[kind]
        examples = [x for x in pool if x.get("category", x.get("rule")) == cat][:3] if kind != "grammar" \
            else [x for x in pool if x["rule"] == cat][:3]
        rows = []
        for i, x in enumerate(examples):
            if kind == "grammar":
                u = utt_by_id.get(x["utt"])
                mine = clip(f"focus{n}_{i}.wav", u["start"], u["end"]) if u else ""
                label = f"{esc(x['text'])} → <b>{esc(x['corrected'])}</b>"
                model_text = x["corrected"]
            elif kind == "intent":
                mine = clip(f"focus{n}_{i}.wav", float(x.get("start", 0)), float(x.get("end", 0)))
                label = f"{esc(x['original'])} → <b>{esc(x['candidate'])}</b>"
                model_text = x["candidate"]
            else:
                mine = clip(f"focus{n}_{i}.wav", float(x.get("start", 0)), float(x.get("end", 0)))
                label = (f"<b>{esc(x['word'])}</b> /{esc(x.get('expected_word', x['expected']))}/ "
                         f"· 聞こえた音 /{esc(x.get('heard_word', x['heard']))}/")
                model_text = clean_word(x["word"])
            ref = reference(f"focus{n}_{i}_model.wav", model_text)
            mark = _tutor_mark(r, kind, cat, x.get("word", x.get("original", "")))
            rows.append(f"<div class='example'><p>{label} <small>{float(x.get('start', utt_by_id.get(x.get('utt'), {}).get('start', 0))):.1f}秒"
                        f"{' · ' + esc(mark) if mark else ''}</small></p>"
                        f"<div class='pair'><span>自分</span>{mine or '<small>—</small>'}"
                        f"<span>お手本</span>{ref or '<small>Piper 未設定</small>'}</div></div>")
        if kind == "grammar":
            explain = examples[0].get("message") if examples and examples[0].get("message") else "LanguageTool の指摘です。"
            practice = list(dict.fromkeys(x["corrected"] for x in pool if x["rule"] == cat))[:5]
        else:
            explain = (("音の取り違えで、別の単語に聞こえた可能性があります。" if kind == "intent" else "")
                       + EXPLAIN.get(cat, ""))
            practice = PRACTICE.get(cat, [])[:5]
        practice_html = "".join(f"<li>{esc(p)}</li>" for p in practice)
        practice_ref = reference(f"focus{n}_practice.wav", practice[0]) if practice else ""
        focus_parts.append(f"<li><h3>{esc(KIND_LABEL[kind])} · {esc(cat)} <small>{t['count']}件</small></h3>"
                           f"<p>{esc(explain)}</p>{''.join(rows)}"
                           f"<p class='sub'>練習用の文</p><ol class='practice'>{practice_html}</ol>"
                           f"{('<p><small>1文目のお手本</small></p>' + practice_ref) if practice_ref else ''}</li>")
    focus = "".join(focus_parts) or "<li>2件以上出た同じ種類の指摘はありません。1件だけの検出は誤検出の可能性があるため、重点に入れていません。</li>"

    def table(rows, cols):
        return ("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for _, h in cols) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{esc(row.get(k, ''))}</td>" for k, _ in cols) + "</tr>" for row in rows) +
                "</table>") if rows else "<p><small>該当なし</small></p>"

    pron_rows = []
    for i, e in enumerate(r["pron_errors"]):
        playback = clip(f"pron_{i}.wav", float(e.get("start", 0)), float(e.get("end", 0)))
        mark = _tutor_mark(r, "pron", e["category"], e["word"])
        cells = [e["word"], f"{e['start']:.2f}", e.get("expected", "") or "（なし）", e.get("heard", "") or "（なし）",
                 e["category"], e["conf"], mark]
        pron_rows.append("<tr class='playable' onclick=\"const a=this.querySelector('audio');if(a&&event.target.tagName!=='AUDIO'){a.currentTime=0;a.play()}\">"
                         + "".join(f"<td>{esc(v)}</td>" for v in cells) + f"<td>{playback}</td></tr>")
    pron_table = ("<table><tr>" + "".join(f"<th>{h}</th>" for h in
                  ("単語", "時刻(秒)", "正しい音素", "実際の音素", "分類", "確信度", "講師モード", "再生")) +
                  "</tr>" + "".join(pron_rows) + "</table>") if pron_rows else "<p><small>該当なし</small></p>"
    intent_rows = [{**x, "mark": _tutor_mark(r, "intent", x["category"], x["word"])} for x in r["intent_candidates"]]
    gram_rows = [{**g, "mark": _tutor_mark(r, "grammar", g["rule"], g["original"])} for g in r["grammar"]]

    history = [row for row in read_history() if row.get("id") != sess["id"]]
    prev = next((row for row in reversed(history) if row["id"] < sess["id"]), None)
    flu_rows = []
    for k, label in FLUENCY_LABELS.items():
        now = r["fluency"].get(k, "")
        before, diff = "—", "—"
        if prev and str(prev.get(k, "")).strip():
            try:
                before = float(prev[k]); diff = f"{float(now) - before:+.3g}"
            except (TypeError, ValueError): pass
        flu_rows.append(f"<tr><td>{esc(label)}</td><td>{esc(now)}</td><td>{esc(before)}</td><td>{esc(diff)}</td></tr>")
    flu_table = "<table><tr><th>指標</th><th>今回</th><th>前回</th><th>差</th></tr>" + "".join(flu_rows) + "</table>"
    trend_rows = [row for row in history if row["id"] < sess["id"]][-9:] + \
        [{"id": sess["id"], **{k: r["fluency"].get(k, "") for k in FLUENCY_LABELS}}]
    trend = ""
    for key, color in (("wpm", "#2467aa"), ("pause_ratio", "#b0632a")):
        vals = []
        for row in trend_rows:
            try: vals.append(float(row.get(key, "")))
            except (TypeError, ValueError): pass
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            points = " ".join(f"{20 + i*540/(len(vals)-1):.1f},{110 - (v-lo)*85/span:.1f}" for i, v in enumerate(vals))
            trend += (f"<p><small>{esc(FLUENCY_LABELS[key])}（{lo:g} 〜 {hi:g}）</small></p>"
                      f"<svg role='img' aria-label='{esc(FLUENCY_LABELS[key])}の推移' viewBox='0 0 580 130' style='width:100%;max-height:160px'>"
                      f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3'/></svg>")
    trend += table(trend_rows, [("id", "セッション"), ("wpm", "語/分"), ("pause_ratio", "沈黙の割合")])

    # Transcript: the learner's utterances with marks, plus partner turns by time.
    entries = [(u["start"], "user", f"<small>{u['start']:.1f}–{u['end']:.1f}秒</small> {_utterance_marks(u, r)}")
               for u in r["utterances"]]
    turns_path = path / "turns.json"
    if turns_path.exists():
        for t in json.loads(turns_path.read_text(encoding="utf-8")):
            if t.get("speaker") in ("partner", "tutor"):
                who = "相手" if t["speaker"] == "partner" else "講師"
                entries.append((float(t.get("t", 0)) - 0.001, t["speaker"],
                                f"<small>{float(t.get('t', 0)):.1f}秒 · {who}</small> {esc(t['text'])}"))
    entries.sort(key=lambda e: e[0])
    transcript = "".join(f"<p class='line {who}'>{body}</p>" for _, who, body in entries)
    legend = ("<p class='legend'><mark class='m-pron'>発音</mark> <mark class='m-intent'>言い間違いの可能性</mark> "
              "<mark class='m-grammar'>文法</mark></p>")
    partner = (path / "partner.txt").read_text(encoding="utf-8") if (path / "partner.txt").exists() else ""
    conversation = (path / "conversation.txt").read_text(encoding="utf-8") if (path / "conversation.txt").exists() else ""
    extra_transcript = ""
    if conversation and not turns_path.exists(): extra_transcript = f"<h3>会話ログ</h3><pre>{esc(conversation)}</pre>"
    elif partner: extra_transcript = f"<h3>貼り付けた相手の発言</h3><pre>{esc(partner)}</pre>"

    corrections = [{**t, "kind_label": KIND_LABEL.get(t.get("kind"), t.get("kind", "")),
                    "result": "合格" if t.get("passed") else "不合格"} for t in r.get("tutor_corrections", [])]
    tutor_section = ""
    if sess.get("practice") == "tutor" or corrections:
        tutor_section = (f"<section><details open><summary>講師モードで訂正した項目</summary>"
                         f"{table(corrections, [('turn', 'ターン'), ('kind_label', '種類'), ('category', '分類'), ('original', '元の表現'), ('better', '訂正後'), ('result', '言い直し'), ('source', '判定元')])}"
                         f"<p><small>文脈・言葉選びの指摘は LLM の判断なので参考扱いです。重点3つの集計には入れていません。</small></p></details></section>")
    errors = "".join(f"<li>{esc(e['stage'])}: {esc(e['error'])}</li>" for e in r["stage_errors"])
    prompt = claude_text(r)
    report_css = (ROOT / "report.css").read_text(encoding="utf-8")
    mode = {"separate": "分離モード", "local": "ローカルモード"}.get(sess["mode"], sess["mode"])
    practice = {"post_analysis": "会話後分析", "tutor": "講師モード"}.get(sess.get("practice"), "")
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>英会話練習レポート {esc(sess['id'])}</title><style>{report_css}</style>
<main class="talk-report"><h1>会話の振り返り</h1>
<section class="summary"><h2>1. サマリー</h2><dl><dt>日時</dt><dd>{esc(_session_datetime(sess))}</dd><dt>モード</dt><dd>{esc(mode)}{(' · ' + esc(practice)) if practice else ''}</dd><dt>シナリオ</dt><dd>{esc(sess.get('scenario') or '—')}</dd><dt>会話時間</dt><dd>{int(sess['duration_sec'] // 60)}分{int(sess['duration_sec'] % 60)}秒</dd><dt>発話数</dt><dd>{len(r['utterances'])}</dd><dt>語数</dt><dd>{r['fluency'].get('word_count', 0)}</dd></dl>
<audio controls preload="none" src="audio.wav"></audio></section>
<section><h2>2. 重点3つ</h2><ol class="focus">{focus}</ol><p><small>音声から舌の位置は断定できません。自分の音声とお手本を聞き比べて確かめてください。</small></p></section>
<section><details><summary>3. 発音の詳細（{len(r['pron_errors'])}件）</summary><p><small>行をクリックすると該当部分を再生します。</small></p>{pron_table}</details></section>
<section><details><summary>4. 言い間違いの可能性（{len(r['intent_candidates'])}件）</summary>{table(intent_rows, [('original', '元の文'), ('candidate', '推定した文'), ('category', '置換の分類'), ('delta_logprob', '自然さの差'), ('mark', '講師モード')])}</details></section>
<section><details><summary>5. 文法（{len(r['grammar'])}件）</summary>{table(gram_rows, [('text', '元の文'), ('corrected', '修正案'), ('rule', '規則名'), ('message', '説明'), ('mark', '講師モード')])}</details></section>
<section><h2>6. 流暢さ</h2>{flu_table}<details><summary>過去のセッションとの推移</summary>{trend}</details></section>
{tutor_section}
<section><details open><summary>7. 文字起こし全文</summary>{legend}{transcript or '<p><small>文字起こしはありません</small></p>'}{extra_transcript}</details></section>
<section data-export-only><h2>8. Claude に貼る用テキスト</h2><button onclick="navigator.clipboard.writeText(document.getElementById('prompt').value).then(()=>this.textContent='コピーしました')">コピー</button><textarea id="prompt" readonly>{esc(prompt)}</textarea></section>
<section><details><summary>判定できなかった項目（{len(r['stage_errors'])}件）</summary><ul>{errors}</ul></details></section></main></html>'''
    (path / "report.html").write_text(page, encoding="utf-8")
    return page


def list_sessions():
    return sorted([p for p in SESSIONS.glob("*/analysis.json")], reverse=True) if SESSIONS.exists() else []


# ---------------------------------------------------------------- export

def export_cards(session_id):
    """One card per finding: front = Japanese note, back = correct English + IPA.
    Fields follow English Express cards (en, ja, ex, exJa, topic)."""
    f = SESSIONS / Path(session_id).name / "analysis.json"
    if not f.exists(): raise FileNotFoundError("分析結果がありません")
    r = json.loads(f.read_text(encoding="utf-8"))
    sid = r["session"]["id"]
    top_other = any(t["kind"] == "pron" and t["category"] == "OTHER" for t in r["top3"])

    def ipa_of(text):
        try: return " ".join("".join(p) for p in phonemize_words(re.findall(r"[A-Za-z']+", text)) if p)
        except Exception: return ""

    cards, seen = [], set()

    def add(key, en, ja, ex, exja, cat):
        if key in seen or not en.strip(): return
        seen.add(key)
        cards.append({"en": en[:80], "ja": ja[:80], "ex": ex[:240], "exJa": exja[:240],
                      "topic": f"英会話/{cat}/{sid}"[:80], "created": 0})

    utt_text = {u["id"]: u["text"] for u in r["utterances"]}
    for e in r["pron_errors"]:
        if e["category"] == "OTHER" and not top_other: continue
        word = clean_word(e["word"])
        expected = e.get("expected_word") or ipa_of(word)
        add(("pron", e["category"], word), word,
            f"【発音 {e['category']}】{word} — {EXPLAIN.get(e['category'], '')}",
            utt_text.get(e["utt"], word), f"IPA /{expected}/ · 聞こえた音 /{e.get('heard_word', e['heard'])}/",
            e["category"])
    for x in r["intent_candidates"]:
        add(("intent", x["word"], x["intended"]), x["candidate"],
            f"【言い間違い {x['category']}】'{x['word']}' ではなく '{x['intended']}'",
            f"× {x['original']}\n○ {x['candidate']}", f"IPA {x['intended']} /{x.get('expected', '')}/ · {x['word']} /{x.get('heard', '')}/",
            x["category"])
    for g in r["grammar"]:
        corrected = g.get("corrected") or g["suggestion"]
        add(("grammar", g["rule"], corrected), corrected,
            f"【文法】{g.get('message') or g['rule']}",
            f"× {g['text']}\n○ {corrected}", f"IPA /{ipa_of(corrected)}/", g["rule"])
    EXPORTS.mkdir(exist_ok=True)
    out = EXPORTS / f"{sid}_english_express_cards.json"
    out.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    return out, cards


# ---------------------------------------------------------------- local mode

def piper(text, path, c=None):
    c = c or config()
    voice = c["tts"]["piper_voice"]
    exe = c["tts"]["piper_exe"]
    voice_path = Path(voice)
    if not voice_path.is_absolute(): voice_path = ROOT / voice_path
    if not voice or not voice_path.is_file(): raise RuntimeError("Piper 音声モデルのパスを設定してください")
    bundled_exe = Path(sys.executable).parent / ("piper.exe" if os.name == "nt" else "piper")
    found = shutil.which(exe) or (exe if Path(exe).is_file() else None) or (str(bundled_exe) if bundled_exe.is_file() else None)
    if not found: raise RuntimeError("Piper 実行ファイルが見つかりません")
    subprocess.run([found, "--model", str(voice_path), "--output_file", str(path)],
                   input=text.encode("utf-8"), capture_output=True, timeout=90, check=True)
    return path


def parse_tutor_json(content):
    """Small models often wrap JSON in code fences or add text around it.
    Anything unreadable means: no context issue, and never read JSON aloud."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced: text = fenced.group(1).strip()
    candidates = [text]
    brace = re.search(r"\{.*\}", text, re.S)
    if brace: candidates.append(brace.group(0))
    for t in candidates:
        try:
            obj = json.loads(t)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            reply = str(obj.get("reply") or "").strip()
            issue = obj.get("context_issue")
            if not (isinstance(issue, dict) and issue.get("original") and issue.get("better")):
                issue = None
            return reply, issue
    if "{" in content or "reply" in content:
        return "", None
    return content.strip(), None


def ollama_reply(messages, scenario, c=None, tutor=False):
    c = c or config()
    model = ollama_local_preflight(c)
    prompt = scenario["system"] + f" Reply in English, in at most {c['local_mode']['max_reply_words']} words."
    body = {"model": model, "stream": False,
            "options": {"num_predict": int(c["local_mode"]["max_reply_words"]) * 3 + 80}}
    if tutor:
        prompt += (' Return only a JSON object with keys "reply" and "context_issue". '
                   '"reply" is your next conversational message. '
                   '"context_issue" is null or {"original":"...","better":"...","reason":"..."} '
                   "about the learner's last message. Only comment on context or word choice. "
                   "Never judge pronunciation or grammar.")
        body["format"] = "json"
    body["messages"] = [{"role": "system", "content": prompt}] + messages
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = json.loads(response.read())
    content = raw.get("message", {}).get("content", "").strip()
    if tutor:
        reply, issue = parse_tutor_json(content)
        return reply or "I see. Could you tell me a little more?", issue
    return content, None


def ollama_local_preflight(c=None):
    """Fail closed before sending conversation text to an Ollama server."""
    c = c or config()
    model = c["local_mode"]["ollama_model"].strip()
    if not model:
        raise RuntimeError("設定タブで Ollama のモデル名を指定してください")
    if model.endswith(":cloud") or "-cloud" in model:
        raise RuntimeError("クラウドモデルは使用できません")
    server_config = Path.home() / ".ollama" / "server.json"
    try:
        settings = json.loads(server_config.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        settings = {}
    if settings.get("disable_ollama_cloud") is not True:
        raise RuntimeError("Ollama の server.json で disable_ollama_cloud=true にし、Ollama を再起動してください")
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
        tags = json.loads(response.read()).get("models", [])
    found = next((m for m in tags if m.get("name") == model or
                  (":" not in model and m.get("name") == model + ":latest")), None)
    if not found or int(found.get("size", 0)) <= 0:
        raise RuntimeError(f"ローカルモデル「{model}」が見つかりません。ollama pull {model} を実行してください")
    return model


def load_asr(model_name, compute_type="int8"):
    from faster_whisper import WhisperModel
    with _load_lock:
        if model_name not in _asr:
            _asr[model_name] = WhisperModel(model_name, device="cpu",
                                            compute_type=compute_type, local_files_only=True)
    return _asr[model_name]


def warm_up_tutor(c=None):
    """Load the per-turn models once, before the first tutor turn needs them."""
    c = c or config()
    noise = np.random.default_rng(0).normal(0, 0.01, RATE).astype("float32")
    steps = [lambda: speech_spans(noise),
             lambda: load_asr(c["tutor"]["asr_model"], c["asr"]["compute_type"]),
             lambda: recognized_phones(noise, c["phoneme"]["model"]),
             lambda: lexicon(c),
             lambda: sentence_logprob("Hello there.", c["intent"]["lm"]),
             lambda: grammar([{"id": 0, "text": "Hello there."}])]
    for step in steps:
        try: step()
        except Exception: pass


def load_scenarios():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "scenarios").glob("*.json"))}
