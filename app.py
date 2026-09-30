"""Run with `uv run app.py`. All recording and model inference stays on localhost."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import gradio as gr
from ui import (HEADER, EMPTY_REPORT, EMPTY_CHAT, RECORDER, THEME, chat_display, heading,
                model_choices, model_description, model_overview)

from engine import (CATEGORIES, DATA, EXPORTS, RATE, ROOT, Recorder, analyze, claude_text,
                    clean_word, config, export_cards, grammar, intent_candidates, list_sessions,
                    load_scenarios, new_session, ollama_local_preflight, ollama_reply, piper,
                    pronunciation, save_config, speech_spans, transcribe, warm_up_tutor, write_report)


rec = Recorder()
scenarios = load_scenarios()
pool = ThreadPoolExecutor(max_workers=1)
active = {"mode": None}
# A turn and the stop button must not touch the session state at the same time.
local_lock = threading.Lock()


def devices():
    try:
        import sounddevice as sd
        return ["システム既定"] + [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    except Exception:
        return ["システム既定"]


def default_device(choices):
    """The saved microphone is stored by name, because device numbers can change."""
    saved = config()["audio"].get("device", "システム既定")
    return next((d for d in choices if d == saved or d.split(": ", 1)[-1] == saved), "システム既定")


def device_id(label):
    return None if not label or label == "システム既定" else int(str(label).split(":", 1)[0])


def progress_callback(progress):
    return lambda stage, fraction=None: progress(fraction or 0, desc=f"分析中: {stage}")


def save_meta(path, mode, scenario=None, practice=None):
    meta = {"mode": mode, "scenario": scenario, "practice": practice,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "tutor_corrections": []}
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def summary(r):
    return (f"分析完了: {r['session']['duration_sec']}秒 · {len(r['utterances'])}発話 · "
            f"{r['fluency'].get('word_count',0)}語 · 判定保留 {len(r['stage_errors'])}項目")


def report_view(r):
    path = DATA / "sessions" / r["session"]["id"]
    try:
        content = (path / "report.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "<p>レポートがありません</p>"
    if '<main class="talk-report">' not in content:
        content = write_report(path, r)
    # The standalone file uses relative audio; Gradio serves the same files via
    # its explicitly allowed local-file route.
    content = re.sub(r"src=([\"'])([^\"']+\.wav)\1",
                     lambda m: f"src={m.group(1)}/gradio_api/file={quote(str(path / m.group(2)), safe='')}{m.group(1)}",
                     content)
    styles = re.search(r"<style>(.*?)</style>", content, re.S)
    main = re.search(r'<main class="talk-report">(.*?)</main>', content, re.S)
    if main:
        body = re.sub(r'<section data-export-only>.*?</section>', '', main.group(1), flags=re.S)
        content = f'<style>{styles.group(1) if styles else ""}</style><main class="talk-report">{body}</main>'
    return content, claude_text(r), str(path / "report.html"), str(path / "analysis.json"), r["session"]["id"]


# ------------------------------------------------------------------ separate mode

def start_separate(device):
    if active["mode"]: return "別の録音が進行中です"
    path = new_session("separate")
    save_meta(path, "separate")
    try: rec.start(device_id(device))
    except Exception as e: return f"マイクを開始できません: {e}"
    active.update(mode="separate", path=path)
    return "録音中です。外部アプリとの会話にはヘッドホンを使ってください。"


def stop_separate(partner, progress=gr.Progress()):
    keep = (gr.skip(),) * 5
    if active.get("mode") != "separate":
        return ("録音していません",) + keep + (gr.Tabs(selected="separate"),)
    path = active["path"]
    try: rec.stop(path / "audio.wav")
    except Exception as e:
        active.clear(); active["mode"] = None
        return (f"保存に失敗: {e}",) + keep + (gr.Tabs(selected="separate"),)
    active.clear(); active["mode"] = None
    r = analyze(path, progress=progress_callback(progress), partner=partner)
    return (summary(r),) + report_view(r) + (gr.Tabs(selected="report"),)


def reanalyze(session_id, progress=gr.Progress()):
    keep = (gr.skip(),) * 5
    if not session_id: return ("履歴からセッションを選んでください",) + keep
    path = DATA / "sessions" / Path(session_id).name
    if not (path / "audio.wav").exists():
        return ("録音がありません",) + keep
    partner_path = path / "partner.txt"
    partner = partner_path.read_text(encoding="utf-8") if partner_path.exists() else ""
    r = analyze(path, progress=progress_callback(progress), partner=partner)
    return (summary(r),) + report_view(r)


# ------------------------------------------------------------------ local mode

def log_lines():
    return "\n".join(active.get("log", []))


def add_turn(speaker, text, t=None):
    """Timed turns let the report place the partner's lines in the transcript."""
    t = rec.seconds() if t is None else t
    active["turns"].append({"speaker": speaker, "text": text, "t": round(t, 2)})
    (active["path"] / "turns.json").write_text(json.dumps(active["turns"], ensure_ascii=False, indent=1), encoding="utf-8")


def speak(text, name):
    """Synthesize with Piper. While the voice plays, the microphone is not
    treated as the learner's turn (it would be the partner's voice)."""
    try:
        out = piper(text, active["path"] / name)
    except Exception:
        return None
    try:
        import soundfile as sf
        active["ignore_until"] = rec.seconds() + sf.info(str(out)).duration + 0.3
    except Exception:
        pass
    return str(out)


def local_start(scenario_id, practice, strength, device):
    if active["mode"]: return "別の録音が進行中です", "", None
    try:
        ollama_local_preflight()
    except Exception as e:
        return f"ローカル会話を始められません: {e}", "", None
    scenario = scenarios[scenario_id]
    path = new_session("local", scenario_id)
    meta = save_meta(path, "local", scenario.get("name", scenario_id), practice)
    active.update(mode="local", path=path, scenario=scenario, practice=practice, strength=strength,
                  meta=meta, log=[], turns=[], messages=[], pending=None, future=None,
                  last_categories=[], ignore_until=0.0)
    try:
        audio = str(piper(scenario["opening"], path / "opening.wav"))
    except Exception as e:
        active.clear(); active["mode"] = None
        return f"Piper を準備できません: {e}", "", None
    try: rec.start(device_id(device))
    except Exception as e:
        active.clear(); active["mode"] = None
        return f"マイクを開始できません: {e}", "", None
    if practice == "tutor":
        threading.Thread(target=warm_up_tutor, daemon=True).start()
    try:
        import soundfile as sf
        active["ignore_until"] = sf.info(audio).duration + 0.3
    except Exception:
        pass
    active["log"].append("相手: " + scenario["opening"])
    active["messages"].append({"role": "assistant", "content": scenario["opening"]})
    add_turn("partner", scenario["opening"], 0.0)
    return "録音中 · 話し終わると相手が答えます", log_lines(), audio


def correction_choice(pron, intents, gram, context, strength, focus, recent,
                      no_repeat=3, pron_categories=None):
    """Priority: slip of the tongue > pronunciation > grammar > context (LLM, reference).
    At most one item. The same category is not corrected in `no_repeat` turns in a row."""
    pron_categories = set(pron_categories or CATEGORIES)
    candidates = []
    if strength in ("focus", "pron", "all"):
        candidates += [{"kind": "intent", "category": x["category"], "original": x["word"],
                        "better": x["intended"], "source": "音素・辞書・言語モデル"} for x in intents]
        candidates += [{"kind": "pron", "category": x["category"], "original": x["word"],
                        "better": clean_word(x["word"]), "source": "音素判定"}
                       for x in pron if x["category"] in pron_categories]
    if strength in ("focus", "grammar", "all"):
        candidates += [{"kind": "grammar", "category": x["rule"], "original": x["original"],
                        "better": x["suggestion"], "source": "LanguageTool"} for x in gram]
        if isinstance(context, dict) and context.get("original") and context.get("better"):
            candidates.append({"kind": "context", "category": "CONTEXT", "original": context["original"],
                               "better": context["better"], "reason": context.get("reason", ""),
                               "source": "LLM（参考）"})
    window = max(0, int(no_repeat) - 1)
    for item in candidates:
        if strength == "focus" and focus and item["category"] not in focus: continue
        if window and len(recent) >= window and all(x == item["category"] for x in recent[-window:]): continue
        return item
    return None


TIPS = {"R_L": "Put your tongue behind your top teeth for L, and do not touch anything for R.",
        "TH": "Put your tongue lightly between your teeth.",
        "V_B": "Touch your lower lip with your upper teeth.",
        "VOWEL": "Change how wide you open your mouth.",
        "EPENTHESIS": "Stop at the last consonant without adding a vowel.",
        "F_H": "For F, touch your lower lip with your upper teeth."}


def correction_sentence(item, sentence):
    if item["kind"] in ("pron", "intent"):
        if item["kind"] == "intent":
            sentence = re.sub(r"\b" + re.escape(item["original"]) + r"\b", item["better"],
                              sentence, count=1, flags=re.I)
        return (f"Quick note: I think you meant '{item['better']}'. "
                f"{TIPS.get(item['category'], 'Listen to the target sound.')} "
                f"Can you say '{sentence}'?")
    return f"Quick note: '{item['original']}' should be '{item['better']}'. Can you say that again?"


def retry_passed(item, utterance, segments, clip, c):
    """Pronunciation: phoneme check of the target word. Grammar/context: the
    corrected phrase is present and LanguageTool no longer raises the rule."""
    if item["kind"] in ("pron", "intent"):
        word = clean_word(item["better"])
        try:
            ww = [{"w": w["w"], "start": w["start"], "end": w["end"]} for seg in segments for w in seg["words"]]
            u = [{"id": 0, "start": 0, "end": len(clip) / RATE, "words": ww, "text": utterance}]
            errs = pronunciation(u, clip, c, c["tutor"]["min_confidence"])
            matching = [w for w in ww if clean_word(w["w"]) == word]
            return (bool(matching) and
                    all(w.get("conf", 0) >= c["tutor"]["min_confidence"] for w in matching) and
                    not any(e["category"] == item["category"] and clean_word(e["word"]) == word for e in errs))
        except Exception:
            return False
    if item["better"].lower() not in utterance.lower():
        return False
    if item["kind"] == "context":
        return True
    try:
        return not any(g["rule"] == item["category"] for g in grammar([{"id": 0, "text": utterance}]))
    except Exception:
        return False


def save_correction(item):
    record = {k: item.get(k) for k in ("turn", "kind", "category", "original", "better", "passed",
                                       "attempts", "source", "reason")}
    active["meta"]["tutor_corrections"].append(record)
    (active["path"] / "meta.json").write_text(json.dumps(active["meta"], ensure_ascii=False, indent=2), encoding="utf-8")


def latest_focus():
    for f in list_sessions():
        try: return {x["category"] for x in json.loads(f.read_text(encoding="utf-8"))["top3"]}
        except Exception: continue
    return set()


def local_poll():
    if active.get("mode") != "local":
        yield gr.skip(), gr.skip(), gr.skip()
        return
    if not local_lock.acquire(blocking=False):
        yield gr.skip(), gr.skip(), gr.skip()
        return
    try:
        if active.get("mode") == "local":
            yield from local_turn()
        else:
            yield gr.skip(), gr.skip(), gr.skip()
    finally:
        local_lock.release()


def local_turn():
    c = config()
    y = rec.snapshot()
    status = rec.status()
    now = len(y) / RATE
    if now < 1.0:
        yield status, gr.skip(), gr.skip(); return
    turn_silence = float(c["local_mode"]["turn_silence_sec"])
    floor = max(rec.last_turn_end, active.get("ignore_until", 0.0))
    offset = max(0.0, floor - 0.2)
    try:
        spans = [{"start": s["start"] + offset, "end": s["end"] + offset}
                 for s in speech_spans(y[round(offset * RATE):], turn_silence)]
    except Exception as e:
        yield "発話の検出を保留: " + str(e), gr.skip(), gr.skip(); return
    # A turn ends only after `turn_silence` seconds of silence (longer than the
    # analysis split), so a short thinking pause does not cut the answer.
    ready = [s for s in spans if s["end"] > floor + 0.05 and now - s["end"] >= turn_silence]
    if not ready:
        yield status, gr.skip(), gr.skip(); return
    start, end = max(ready[0]["start"], floor), ready[-1]["end"]
    rec.last_turn_end = end
    clip = y[round(start * RATE):round(end * RATE)]
    tutor = active["practice"] == "tutor"
    yield "文字起こし中…", gr.skip(), gr.skip()
    try:
        segments = transcribe(clip, c["tutor"]["asr_model"] if tutor else c["asr"]["model"], c["asr"]["compute_type"])
    except Exception as e:
        yield "文字起こしを保留: " + str(e), gr.skip(), gr.skip(); return
    utterance = " ".join(seg["text"] for seg in segments).strip()
    if not utterance:
        yield status, gr.skip(), gr.skip(); return
    active["log"].append("自分: " + utterance)
    add_turn("user", utterance, start)

    item = active["pending"]
    if item:
        # Re-say after a correction.
        yield "言い直しを確認中…", log_lines(), gr.skip()
        item["attempts"] += 1
        passed = retry_passed(item, utterance, segments, clip, c)
        if not passed and item["attempts"] < int(c["tutor"]["max_retries"]):
            prompt = correction_sentence(item, item["sentence"])
            active["log"].append("判定: もう一度")
            active["log"].append("講師: " + prompt)
            add_turn("tutor", prompt)
            yield status, log_lines(), speak(prompt, f"correction_{len(active['log'])}.wav") or gr.skip()
            return
        item["passed"] = passed
        save_correction(item)
        active["pending"] = None
        lead = "Great!" if passed else "No problem, let's keep going."
        active["log"].append("判定: " + ("合格" if passed else "不合格"))
        active["log"].append("講師: " + lead)
        add_turn("tutor", lead)
        yield "相手の返答を待っています…", log_lines(), gr.skip()
        reply = wait_reply(active.pop("future", None))
        active["future"] = None
        yield status, log_lines(), say_reply(reply, lead)
        return

    active["messages"].append({"role": "user", "content": utterance})
    if not tutor:
        yield "相手の返答を待っています…", log_lines(), gr.skip()
        try: reply, _ = ollama_reply(active["messages"], active["scenario"], c)
        except Exception as e: reply = f"会話の生成に失敗しました: {e}"
        yield status, log_lines(), say_reply(reply)
        return

    # Tutor turn: the LLM writes the next reply while this utterance is judged.
    future = pool.submit(ollama_reply, list(active["messages"]), active["scenario"], c, True)
    yield "判定中…", log_lines(), gr.skip()
    thr = c["tutor"]["min_confidence"]
    u = [{"id": 0, "start": 0, "end": len(clip) / RATE, "text": utterance,
          "words": [{"w": w["w"], "start": w["start"], "end": w["end"]} for seg in segments for w in seg["words"]]}]
    try: pron = pronunciation(u, clip, c, thr)
    except Exception: pron = []
    try: gram = grammar(u)
    except Exception: gram = []
    try: intents = intent_candidates(u, c)
    except Exception: intents = []
    focus = latest_focus() if active["strength"] == "focus" else set()
    args = dict(no_repeat=c["tutor"]["no_repeat_turns"], pron_categories=c["tutor"]["pron_categories"])
    item = correction_choice(pron, intents, gram, None, active["strength"], focus, active["last_categories"], **args)
    reply = None
    if not item:
        yield "相手の返答を待っています…", log_lines(), gr.skip()
        try: reply, context = future.result(timeout=120)
        except Exception as e: reply, context = f"会話の生成に失敗しました: {e}", None
        item = correction_choice([], [], [], context, active["strength"], focus, active["last_categories"], **args)
    active["last_categories"].append(item["category"] if item else None)
    if not item:
        yield status, log_lines(), say_reply(reply)
        return
    item.update(turn=sum(m["role"] == "user" for m in active["messages"]), attempts=0, passed=None, sentence=utterance)
    active["pending"] = item
    active["future"] = future
    sentence = correction_sentence(item, utterance)
    active["log"].append("講師: " + sentence)
    add_turn("tutor", sentence)
    yield status, log_lines(), speak(sentence, f"correction_{len(active['log'])}.wav") or gr.skip()


def wait_reply(future):
    if not future: return "Let's keep going. What would you like to talk about next?"
    try: reply, _ = future.result(timeout=120)
    except Exception as e: reply = f"会話の生成に失敗しました: {e}"
    return reply


def say_reply(reply, lead=""):
    failed = reply.startswith("会話の生成に失敗しました")
    active["log"].append("相手: " + reply)
    if not failed:
        active["messages"].append({"role": "assistant", "content": reply})
        add_turn("partner", reply)
    text = f"{lead} {reply}".strip() if not failed else lead
    return (speak(text, f"reply_{len(active['log'])}.wav") if text else None) or gr.skip()


def local_stop(progress=gr.Progress()):
    keep = (gr.skip(),) * 5
    if active.get("mode") != "local":
        return ("録音していません", gr.skip()) + keep + (gr.Tabs(selected="local"),)
    with local_lock:
        path = active["path"]
        log = log_lines()
        try: rec.stop(path / "audio.wav")
        except Exception as e:
            return (f"保存に失敗: {e}", log) + keep + (gr.Tabs(selected="local"),)
        if active.get("pending"):
            item = active["pending"]; item["passed"] = None
            save_correction(item)
        (path / "conversation.txt").write_text(log, encoding="utf-8")
        active.clear(); active["mode"] = None
    r = analyze(path, progress=progress_callback(progress))
    return (summary(r), log) + report_view(r) + (gr.Tabs(selected="report"),)


# ------------------------------------------------------------------ settings / history

def settings_save(asr, phoneme_model, intent_lm, conf, delta, silence, turn_silence, denoise, device,
                  ollama, exe, voice, tutor_asr, tutor_conf, tutor_cats, tutor_strength, no_repeat):
    if active.get("mode"): return "録音を終了してから設定を保存してください。"
    c = config()
    old_models = (c["phoneme"]["model"], c["intent"]["lm"], c["asr"]["model"], c["tutor"]["asr_model"])
    c["asr"]["model"] = asr.strip()
    c["phoneme"]["model"] = phoneme_model.strip()
    c["intent"]["lm"] = intent_lm.strip()
    c["phoneme"]["min_confidence"] = float(conf)
    c["intent"]["min_delta_logprob"] = float(delta)
    c["vad"]["min_silence_sec"] = float(silence)
    c["local_mode"]["turn_silence_sec"] = float(turn_silence)
    c["audio"]["denoise"] = bool(denoise)
    c["audio"]["device"] = device.split(": ", 1)[-1] if device and device != "システム既定" else "システム既定"
    c["local_mode"]["ollama_model"] = (ollama or "").strip()
    c["tts"]["piper_exe"] = exe.strip()
    c["tts"]["piper_voice"] = voice.strip()
    c["tutor"]["asr_model"] = tutor_asr.strip()
    c["tutor"]["min_confidence"] = float(tutor_conf)
    c["tutor"]["pron_categories"] = list(tutor_cats or [])
    c["tutor"]["strength"] = tutor_strength
    c["tutor"]["no_repeat_turns"] = int(no_repeat)
    save_config(c)
    note = ""
    if old_models != (c["phoneme"]["model"], c["intent"]["lm"], c["asr"]["model"], c["tutor"]["asr_model"]):
        note = " モデルを変えた場合は、PowerShell で .\\run.ps1 models を実行して保存してください。"
    return "設定を保存しました。次のセッションから反映されます。" + note


FLUENCY_METRICS = {"wpm": "1分あたりの語数", "articulation_rate": "発話速度（無音を除く）",
                   "pause_ratio": "沈黙の割合", "mean_run": "続けて話した語数の平均", "fillers": "つなぎ言葉の回数"}


def history_data():
    rows, categories, fluency_rows = [], [], []
    for f in list_sessions():
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = r["session"]["id"]
        rows.append([sid, r["session"]["mode"], r["fluency"].get("wpm"), r["fluency"].get("pause_ratio"),
                     len(r["pron_errors"]), len(r["intent_candidates"]), len(r["grammar"])])
        counts = {}
        for e in r["pron_errors"]:
            counts["発音:" + e["category"]] = counts.get("発音:" + e["category"], 0) + 1
        for e in r["intent_candidates"]:
            counts["言い間違い:" + e["category"]] = counts.get("言い間違い:" + e["category"], 0) + 1
        if r["grammar"]:
            counts["文法"] = len(r["grammar"])
        categories += [{"session": sid, "category": k, "count": n} for k, n in counts.items()]
        fluency_rows.append({"session": sid, **{k: r["fluency"].get(k) for k in FLUENCY_METRICS}})
    return rows, categories, list(reversed(fluency_rows))


def history_view(metric="wpm"):
    import pandas as pd
    rows, categories, fluency_rows = history_data()
    choices = [r[0] for r in rows]
    metric = metric if metric in FLUENCY_METRICS else "wpm"
    flu = pd.DataFrame([{"session": x["session"], "value": x[metric]} for x in fluency_rows],
                       columns=["session", "value"])
    cats = pd.DataFrame(sorted(categories, key=lambda x: x["session"]), columns=["session", "category", "count"])
    return (rows, gr.Dropdown(choices=choices, value=choices[0] if choices else None),
            gr.LinePlot(value=flu, title=FLUENCY_METRICS[metric], y_title=FLUENCY_METRICS[metric]), cats)


def fluency_plot(metric):
    import pandas as pd
    _, _, fluency_rows = history_data()
    flu = pd.DataFrame([{"session": x["session"], "value": x[metric]} for x in fluency_rows],
                       columns=["session", "value"])
    return gr.LinePlot(value=flu, title=FLUENCY_METRICS[metric], y_title=FLUENCY_METRICS[metric])


def selected_report(session_id):
    if not session_id: return EMPTY_REPORT, "", None, None, None
    f = DATA / "sessions" / Path(session_id).name / "analysis.json"
    if not f.exists(): return "<p>分析結果がありません</p>", "", None, None, None
    return report_view(json.loads(f.read_text(encoding="utf-8")))


def export_current_cards(session_id):
    # The session id comes from state: gr.File values point at Gradio's cache
    # folder, not at the session folder.
    if not session_id: return None, "先にレポートを開いてください"
    try:
        out, cards = export_cards(session_id)
    except FileNotFoundError as e:
        return None, str(e)
    if not cards:
        return None, "書き出す指摘がありませんでした。"
    return str(out), (f"{len(cards)}枚のカードを書き出しました（1件の指摘につき1枚）。"
                      "English-Express-with-import.html の「英会話カードを取り込む」から読み込めます。")


def check_local_model(model):
    """Read localhost inventory only. Never start a model download."""
    if not model: return "モデルを選んでください。"
    if model.endswith(":cloud") or "-cloud" in model:
        return "クラウドモデルは利用できません。ローカルモデルを選んでください。"
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            items = json.load(response).get("models", [])
    except Exception:
        exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if not shutil.which("ollama") and not exe.is_file():
            return "Ollama 本体が見つかりません。①の公式リンクからインストールし、PowerShell を開き直してから導入コマンドを実行してください。"
        return "Ollama 本体は見つかりましたが、起動していません。スタートメニューから Ollama を起動して、もう一度確認してください。"
    names = {m.get("name", "") for m in items if m.get("size", 0) > 0}
    if model not in names and model + ":latest" not in names:
        return "Ollama は起動しています。選択したモデルは未導入です。上の導入コマンドを実行してください。"
    server = Path.home() / ".ollama" / "server.json"
    try: cloud_disabled = json.loads(server.read_text(encoding="utf-8")).get("disable_ollama_cloud") is True
    except (FileNotFoundError, ValueError): cloud_disabled = False
    if not cloud_disabled: return "モデルは導入済みです。「ローカル実行の準備」でクラウド無効化の設定をしてください。"
    return "モデルの保存とクラウド無効化の設定を確認しました。設定変更後は Ollama を再起動してください。"


def local_start_view(*args):
    status, log, audio = local_start(*args)
    return status, chat_display(log), audio


def local_poll_view():
    for status, log, audio in local_poll():
        yield status, (chat_display(log) if isinstance(log, str) else log), audio


def local_stop_view(progress=gr.Progress()):
    result = list(local_stop(progress))
    if isinstance(result[1], str): result[1] = chat_display(result[1])
    return tuple(result)


# ------------------------------------------------------------------ layout

CFG = config()
MICS = devices()
MIC_DEFAULT = default_device(MICS)

with gr.Blocks(title="English Talk · 英会話の練習室", analytics_enabled=False) as demo:
    current_session = gr.State(None)
    gr.HTML(HEADER)
    with gr.Tabs(selected="separate", elem_id="main-tabs") as tabs:
        with gr.Tab("ローカル", id="local"):
            gr.HTML(heading("LOCAL CONVERSATION", "今日は、どんな話をしよう。", "PC 内の AI と英語で会話。場面を選んで、自分のペースで練習しましょう。"))
            with gr.Row():
                with gr.Column(scale=4, min_width=300, elem_classes="surface"):
                    gr.Markdown("## 会話を準備する")
                    local_scenario = gr.Dropdown(choices=[(v["name"], k) for k, v in scenarios.items()],
                                                 value="interview" if "interview" in scenarios else next(iter(scenarios)),
                                                 label="話したい場面")
                    practice = gr.Radio(choices=[("会話後分析（会話に集中）", "post_analysis"), ("講師モード（その場で訂正）", "tutor")],
                                        value=CFG["local_mode"]["practice"], label="練習モード",
                                        info="会話後分析：会話中は訂正せず、終了後にまとめて分析。講師モード：1ターンに最大1つ訂正し、言い直しを確認。")
                    with gr.Accordion("マイク・訂正の強さ", open=False):
                        strength = gr.Dropdown(choices=[("重点分類のみ（前回の重点3つ）", "focus"), ("発音のみ（言い間違い・発音）", "pron"),
                                                        ("文法のみ（文法・文脈）", "grammar"), ("すべて", "all")],
                                               value=CFG["tutor"]["strength"], label="訂正の強さ（講師モードのみ）")
                        local_device = gr.Dropdown(choices=MICS, value=MIC_DEFAULT, label="使うマイク")
                    start_local = gr.Button("会話を始める →", variant="primary")
                    end_local = gr.Button("終了して振り返る")
                    local_status = gr.Textbox(value="待機中 · ヘッドホンを着けて始めましょう", show_label=False, interactive=False, elem_classes="status-box")
                    gr.Markdown("初回は「設定」で会話モデルを選び、導入してください。", elem_classes="subtle-note")
                with gr.Column(scale=7, min_width=300, elem_classes="surface"):
                    gr.Markdown("## 会話")
                    local_log = gr.HTML(value=EMPTY_CHAT, label="会話ログ", elem_classes="chat-panel")
                    local_audio = gr.Audio(label="相手の声", autoplay=True, interactive=False)
        with gr.Tab("分離", id="separate"):
            gr.HTML(heading("RECORD & REFLECT", "いつもの会話を、学びに。", "外部アプリで話しながら録音。終わったら、この PC で振り返ります。"))
            with gr.Row():
                with gr.Column(scale=7, min_width=300, elem_classes="surface"):
                    gr.HTML(RECORDER)
                    separate_device = gr.Dropdown(choices=MICS, value=MIC_DEFAULT, label="使うマイク")
                    with gr.Row():
                        start_sep = gr.Button("録音を始める", variant="primary")
                        end_sep = gr.Button("終了して振り返る")
                    sep_status = gr.Textbox(value="待機中 · まだ録音していません", label="録音時間・入力レベル", show_label=False, interactive=False, elem_classes="status-box")
                with gr.Column(scale=4, min_width=260):
                    gr.HTML('<aside class="guide"><h2>練習の流れ</h2><ol><li><strong>ヘッドホンを着ける（必須）</strong><br>相手の声が録音に混ざるのを防ぎます。</li><li><strong>録音して、英語で会話</strong><br>会話相手のアプリは別の画面で開きます。</li><li><strong>終了して振り返る</strong><br>発音・文法・流暢さを確認。次の重点を見つけます。</li></ol><small>Windows のサウンド設定でマイクの「排他モード」が有効だと、外部アプリと同時に録音できません。</small></aside>')
                    with gr.Accordion("相手の発言も残す（任意）", open=False):
                        partner = gr.Textbox(label="相手の会話履歴", lines=6, placeholder="外部アプリの会話履歴を貼り付けてください。")
        with gr.Tab("レポート", id="report"):
            gr.HTML(heading("YOUR REFLECTION", "小さな気づきを、次の会話へ。", "まずは重点から。音声を聞き直しながら、自分の変化を確かめましょう。"))
            report_status = gr.Textbox(value="会話の終了後に分析します", label="分析状態", show_label=False, interactive=False, elem_classes="status-box")
            report_html = gr.HTML(value=EMPTY_REPORT, label="レポート")
            with gr.Accordion("8. Claude に貼る用テキスト・レポートの保存", open=False):
                copy_text = gr.Textbox(label="Claude に貼る用テキスト", lines=8, buttons=["copy"], info="コピーして、普段の Claude のチャットにご自身で貼り付けてください。")
                with gr.Row():
                    report_file = gr.File(label="HTML レポート")
                    analysis_file = gr.File(label="分析データ · JSON")
            with gr.Accordion("English Express に書き出す", open=False):
                gr.Markdown("表示中のレポートの指摘を、1件につき1枚の復習カードにします（表：日本語の注意点、裏：正しい英文と IPA）。")
                export_button = gr.Button("English Express に書き出す")
                export_file = gr.File(label="復習カード · JSON")
                export_status = gr.Textbox(label="書き出し状態", interactive=False)
        with gr.Tab("履歴", id="history"):
            gr.HTML(heading("YOUR PRACTICE", "練習の積み重ねを、見える形に。", "過去の会話を開くことも、同じ録音をもう一度分析することもできます。"))
            with gr.Column(elem_classes="surface"):
                with gr.Row():
                    hist_select = gr.Dropdown(choices=[], value=None, label="保存した会話", scale=3)
                    refresh = gr.Button("履歴を更新", scale=1)
                with gr.Row():
                    show_report = gr.Button("レポートを開く", variant="primary")
                    rerun = gr.Button("録音を再分析する（今の設定で）")
                hist_table = gr.Dataframe(headers=["セッション", "モード", "語/分", "沈黙の割合", "発音の候補", "言い間違いの候補", "文法の指摘"], interactive=False)
            with gr.Row():
                with gr.Column():
                    metric = gr.Dropdown(choices=[(v, k) for k, v in FLUENCY_METRICS.items()], value="wpm", label="流暢さの指標")
                    fluency_chart = gr.LinePlot(x="session", y="value", title=FLUENCY_METRICS["wpm"], x_label_angle=45)
                category_chart = gr.LinePlot(x="session", y="count", color="category", title="分類ごとの件数の推移", x_label_angle=45)
            gr.Markdown("話す速さや検出件数だけでは、会話の良し悪しを判断できません。音声とあわせて振り返ってください。", elem_classes="subtle-note")
        with gr.Tab("設定", id="settings"):
            gr.HTML(heading("MAKE IT YOURS", "自分に合う会話相手を選ぶ。", "Ollama は、この PC で AI を動かすためのソフトです。モデル名は、使う AI の種類とサイズを表します。"))
            gr.HTML(model_overview())
            initial_model = CFG["local_mode"]["ollama_model"] or "llama3.2:3b"
            with gr.Row():
                with gr.Column(scale=7, min_width=300, elem_classes="surface"):
                    gr.Markdown("## 会話モデル")
                    ollama = gr.Dropdown(choices=model_choices(), value=initial_model, allow_custom_value=True, label="Ollama のモデル", info="候補を選ぶか、導入済みのモデル名を入力できます。")
                    model_info = gr.Markdown(model_description(initial_model)[0], elem_classes="model-detail")
                    gr.Markdown("**① Ollama 本体を導入**　[Windows 版をダウンロード](https://ollama.com/download/windows)\n\n**② PowerShell を開き直す**　インストール前から開いている画面では `ollama` が認識されないことがあります。\n\n**③ 下のコマンドでモデルを導入**　完了したら「設定を保存」します。", elem_classes="subtle-note")
                    model_command = gr.Textbox(value=model_description(initial_model)[1], label="導入コマンド · PowerShell で実行", interactive=False, buttons=["copy"])
                    with gr.Accordion("「ollama が認識されません」と出たら", open=False):
                        gr.Markdown('Ollama 本体をインストール後、PowerShell を開き直してください。通常のインストール先であれば、次のように実行ファイルの場所を直接指定する方法もあります。\n\n```powershell\n& "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" pull llama3.2:3b\n```\n\n候補を変えた場合は、最後のモデル名も変更してください。')
                    model_check = gr.Button("この PC の導入状況を確認")
                    model_check_status = gr.Textbox(label="導入状況", interactive=False, placeholder="確認すると、Ollama とモデルの状態を表示します。")
                with gr.Column(scale=4, min_width=260):
                    gr.HTML('<aside class="guide"><h2>名前の読み方</h2><p><strong>llama3.2 : 3b</strong><br>llama3.2 → モデルの種類・バージョン<br>3b → 約 30 億パラメータ</p><p>パラメータは、学習で調整する数値です。数が大きいほど容量が増える傾向がありますが、会話の質は数だけでは決まりません。</p><p><strong>まずは小さい候補から</strong><br>このアプリでの応答速度は未測定です。短い会話で動作を確かめましょう。</p></aside>')
                    with gr.Accordion("ローカル実行の準備", open=False):
                        gr.Markdown('[Ollama をダウンロード](https://ollama.com/download/windows)\n\n`%USERPROFILE%\\.ollama\\server.json` に `"disable_ollama_cloud": true` を設定し、Ollama を再起動してください。既存の項目があれば追加します。詳しい手順は同梱 README にあります。')
            with gr.Accordion("マイク・音声・分析の詳細設定", open=False):
                with gr.Row():
                    with gr.Column():
                        mic_default = gr.Dropdown(choices=MICS, value=MIC_DEFAULT, label="既定のマイク")
                        asr = gr.Textbox(label="Whisper モデル（会話後の分析）", value=CFG["asr"]["model"])
                        phoneme_model = gr.Textbox(label="音素認識モデル", value=CFG["phoneme"]["model"])
                        intent_lm = gr.Textbox(label="言い間違い推定の言語モデル", value=CFG["intent"]["lm"])
                        denoise = gr.Checkbox(value=CFG["audio"]["denoise"], label="ノイズ除去（DeepFilterNet・任意導入。README 参照）")
                        piper_exe = gr.Textbox(value=CFG["tts"]["piper_exe"], label="Piper 実行ファイル")
                        piper_voice = gr.Textbox(value=CFG["tts"]["piper_voice"], label="Piper 音声モデル .onnx のパス")
                    with gr.Column():
                        conf = gr.Slider(0.1, 0.95, value=CFG["phoneme"]["min_confidence"], step=0.05, label="音素の最低確信度（会話後の分析）")
                        delta = gr.Number(value=CFG["intent"]["min_delta_logprob"], label="言い間違い候補の自然さの差（対数尤度）")
                        silence = gr.Slider(0.25, 1.5, value=CFG["vad"]["min_silence_sec"], step=0.05, label="分析で発話を区切る無音（秒）")
                        turn_silence = gr.Slider(0.6, 3.0, value=CFG["local_mode"]["turn_silence_sec"], step=0.1, label="ローカル会話で話し終わりとみなす無音（秒）")
                        gr.Markdown("**講師モード**")
                        tutor_asr = gr.Textbox(label="Whisper モデル（講師モードの簡易判定）", value=CFG["tutor"]["asr_model"])
                        tutor_conf = gr.Slider(0.5, 0.95, value=CFG["tutor"]["min_confidence"], step=0.05, label="音素の最低確信度（講師モード）")
                        tutor_cats = gr.CheckboxGroup(choices=CATEGORIES, value=CFG["tutor"]["pron_categories"], label="会話中に訂正する発音の分類")
                        tutor_strength = gr.Dropdown(choices=[("重点分類のみ", "focus"), ("発音のみ", "pron"), ("文法のみ", "grammar"), ("すべて", "all")],
                                                     value=CFG["tutor"]["strength"], label="訂正の強さ（既定）")
                        no_repeat = gr.Number(value=CFG["tutor"]["no_repeat_turns"], precision=0, label="同じ分類を続けて指摘しないターン数")
            with gr.Row():
                save_settings_btn = gr.Button("設定を保存", variant="primary", scale=1)
                settings_status = gr.Textbox(label="設定状態", show_label=False, interactive=False, scale=3, placeholder="選んだモデルは、保存してから会話に使われます。")
    gr.HTML('<footer class="app-footer">English Talk · 話して、気づいて、また話す。</footer>')

    report_outputs = [report_html, copy_text, report_file, analysis_file, current_session]
    timer = gr.Timer(1.0)
    timer.tick(lambda: rec.status() if active.get("mode") == "separate" else gr.skip(), outputs=sep_status)
    timer.tick(local_poll_view, outputs=[local_status, local_log, local_audio])
    start_sep.click(start_separate, inputs=separate_device, outputs=sep_status)
    end_sep.click(stop_separate, inputs=partner, outputs=[report_status] + report_outputs + [tabs])
    start_local.click(local_start_view, inputs=[local_scenario, practice, strength, local_device],
                      outputs=[local_status, local_log, local_audio])
    end_local.click(local_stop_view, outputs=[report_status, local_log] + report_outputs + [tabs])
    save_settings_btn.click(settings_save, inputs=[asr, phoneme_model, intent_lm, conf, delta, silence, turn_silence,
                                                   denoise, mic_default, ollama, piper_exe, piper_voice,
                                                   tutor_asr, tutor_conf, tutor_cats, tutor_strength, no_repeat],
                            outputs=settings_status)
    ollama.change(model_description, inputs=ollama, outputs=[model_info, model_command])
    model_check.click(check_local_model, inputs=ollama, outputs=model_check_status)
    refresh.click(history_view, inputs=metric, outputs=[hist_table, hist_select, fluency_chart, category_chart])
    metric.change(fluency_plot, inputs=metric, outputs=fluency_chart)
    show_report.click(selected_report, inputs=hist_select, outputs=report_outputs).then(lambda: gr.Tabs(selected="report"), outputs=tabs)
    rerun.click(reanalyze, inputs=hist_select, outputs=[report_status] + report_outputs).then(lambda: gr.Tabs(selected="report"), outputs=tabs)
    export_button.click(export_current_cards, inputs=current_session, outputs=[export_file, export_status])
    demo.load(history_view, inputs=metric, outputs=[hist_table, hist_select, fluency_chart, category_chart])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1",
                        server_port=int(os.environ.get("ENGLISH_TALK_PORT", "7860")),
                        share=False, inbrowser=os.environ.get("ENGLISH_TALK_OPEN_BROWSER", "1") == "1",
                        allowed_paths=[str(DATA), str(EXPORTS)], theme=THEME,
                        css=(ROOT / "ui.css").read_text(encoding="utf-8"), footer_links=[])
