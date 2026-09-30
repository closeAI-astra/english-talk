import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine


def has_espeak():
    try:
        engine.phonemize_words(["test"])
        return True
    except Exception:
        return False


needs_espeak = pytest.mark.skipif(not has_espeak(), reason="phonemizer/espeak-ng が必要")


def test_alignment_and_category():
    ops = engine.edit_alignment(["ɹ", "aɪ", "t"], ["l", "aɪ", "t"])
    assert ops == [("sub", 0, 0), ("eq", 1, 1), ("eq", 2, 2)]
    assert engine.category("ɹ", "l") == "R_L"
    assert engine.category("θ", "s") == "TH"
    assert engine.category("v", "b") == "V_B"
    assert engine.category("æ", "ɑː") == "VOWEL"
    assert engine.category("", "ə", insertion=True, final=True) == "EPENTHESIS"
    assert engine.category("", "ə", insertion=True, final=False) == "OTHER"


def phones(seq):
    return [{"phone": p, "start": i * .1, "end": i * .1 + .1, "conf": .9} for i, p in enumerate(seq)]


def test_epenthesis_only_after_final_consonant():
    desk = ["d", "ɛ", "s", "k"]
    final = engine.compare_word(desk, phones(desk + ["ʊ"]), .6)
    assert [e["category"] for e in final] == ["EPENTHESIS"]
    middle = engine.compare_word(desk, phones(["d", "ɛ", "ə", "s", "k"]), .6)
    assert [e["category"] for e in middle] == ["OTHER"]
    # "see" ends in a vowel: an extra vowel is not epenthesis.
    see = engine.compare_word(["s", "iː"], phones(["s", "iː", "ə"]), .6)
    assert [e["category"] for e in see] == ["OTHER"]


def test_phone_belongs_to_one_word_by_midpoint():
    heard = [{"phone": "t", "start": .47, "end": .56, "conf": .9}]
    first = engine.phones_in_word(heard, 0, .5)
    second = engine.phones_in_word(heard, .5, 1.0)
    assert (len(first), len(second)) == (0, 1)


@needs_espeak
def test_phonemize_keeps_one_entry_per_word():
    out = engine.phonemize_words(["I", "2026", "think"])
    assert len(out) == 3 and out[1] == [] and out[2] == ["θ", "ɪ", "ŋ", "k"]


@needs_espeak
def test_intent_finds_light_for_right():
    c = engine.config()
    utt = [{"id": 0, "text": "Can you turn on the right?", "start": 0, "end": 2,
            "words": [{"w": w, "start": i * .3, "end": i * .3 + .25}
                      for i, w in enumerate(["Can", "you", "turn", "on", "the", "right?"])]}]
    fake = lambda s: 10.0 if "light" in s else 0.0
    found = engine.intent_candidates(utt, c, logprob=fake)
    hit = [x for x in found if x["word"] == "right"]
    assert hit and hit[0]["intended"] == "light" and hit[0]["category"] == "R_L"
    assert hit[0]["candidate"] == "Can you turn on the light?"
    assert hit[0]["start"] == 1.5


def test_focus_requires_repetition_and_tiebreak():
    pron = [{"category": "TH"}, {"category": "TH"}, {"category": "R_L"}]
    intent = [{"category": "V_B"}, {"category": "V_B"}]
    gram = [{"rule": "SUBJECT_VERB_AGREEMENT"}] * 2
    top = engine.top_three(pron, intent, gram)
    assert [(x["kind"], x["category"]) for x in top] == [
        ("pron", "TH"), ("intent", "V_B"), ("grammar", "SUBJECT_VERB_AGREEMENT")]
    assert all(x["category"] != "R_L" for x in top)
    noise = [{"category": "OTHER", "utt": 0}] * 15
    assert engine.top_three(noise, [], []) == []


def test_fluency_is_time_based():
    words = [{"w": "I", "start": 0.2, "end": 0.4},
             {"w": "um", "start": 0.5, "end": 0.7},
             {"w": "agree", "start": 2.1, "end": 2.6}]
    u = [{"words": words}]
    spans = [{"start": 0.2, "end": 0.8}, {"start": 2.0, "end": 2.8}]
    f = engine.fluency(u, spans, 3.0)
    assert f["wpm"] == 60.0
    assert f["fillers"] == 1
    assert f["pause_ratio"] == 0.4
    assert f["mean_run"] == 1.5


def test_silence_never_becomes_prompt_text():
    import numpy as np
    assert engine.transcribe(np.zeros(16000, dtype="float32")) == []


def test_tutor_json_is_never_read_aloud():
    fenced = '```json\n{"reply": "Nice! What next?", "context_issue": {"original": "I\'m boring", "better": "I\'m bored", "reason": "x"}}\n```'
    reply, issue = engine.parse_tutor_json(fenced)
    assert reply == "Nice! What next?" and issue["better"] == "I'm bored"
    assert engine.parse_tutor_json('Sure: {"reply": "Hi", "context_issue": null}') == ("Hi", None)
    assert engine.parse_tutor_json('{"reply": "Hi", "context_issue": "none"}') == ("Hi", None)
    assert engine.parse_tutor_json('{"reply": broken')[0] == ""
    assert engine.parse_tutor_json("Just text.") == ("Just text.", None)


def sample_result(sid="one"):
    return {"session": {"id": sid, "mode": "local", "practice": "tutor", "scenario": "面接官", "duration_sec": 2.0},
            "utterances": [{"id": 0, "start": 0.0, "end": 1.0, "text": "<script>alert(1)</script> He go to school",
                            "words": [{"w": "He", "start": .2, "end": .3}]}],
            "pron_errors": [{"utt": 0, "word": "think", "start": .1, "end": .3, "category": "TH",
                             "expected": "θ", "heard": "s", "expected_word": "θɪŋk", "heard_word": "sɪŋk", "conf": .9}],
            "intent_candidates": [{"utt": 0, "word": "right", "intended": "light", "category": "R_L", "delta_logprob": 5.0,
                                   "original": "Turn on the right", "candidate": "Turn on the light", "start": .5, "end": .7,
                                   "expected": "laɪt", "heard": "ɹaɪt"}],
            "grammar": [{"utt": 0, "text": "He go to school", "original": "go", "suggestion": "goes",
                         "corrected": "He goes to school", "offset": 29, "length": 2, "rule": "AGREEMENT", "message": "Use goes."}],
            "fluency": {"wpm": 0, "articulation_rate": 0, "pause_ratio": 0, "mean_run": 0, "fillers": 0, "word_count": 0},
            "top3": [], "tutor_corrections": [{"turn": 1, "kind": "grammar", "category": "AGREEMENT", "original": "go",
                                               "better": "goes", "passed": True, "source": "LanguageTool"}],
            "stage_errors": [{"stage": "発音", "error": "model missing"}]}


def test_report_escapes_marks_and_lists_partner(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "DATA", tmp_path)
    path = tmp_path / "sessions" / "one"
    path.mkdir(parents=True)
    (path / "turns.json").write_text(json.dumps([{"speaker": "partner", "text": "Tell me about you.", "t": 0}]), encoding="utf-8")
    r = sample_result()
    page = engine.write_report(path, r)
    assert "&lt;script&gt;" in page
    assert "<script>alert(1)</script>" not in page
    assert "model missing" in page
    assert "Tell me about you." in page
    assert "m-grammar" in page and "訂正済み・合格" in page
    assert "He goes to school" in page
    for n in range(1, 9):
        assert f"{n}. " in page
    engine.update_history(r)
    assert (tmp_path / "history.csv").exists()


def test_export_one_card_per_finding(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(engine, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(engine, "phonemize_words", lambda ws: [["x"] for _ in ws])
    path = tmp_path / "sessions" / "one"
    path.mkdir(parents=True)
    (path / "analysis.json").write_text(json.dumps(sample_result()), encoding="utf-8")
    out, cards = engine.export_cards("one")
    assert out.exists() and len(cards) == 3
    assert {c["topic"] for c in cards} == {"英会話/TH/one", "英会話/R_L/one", "英会話/AGREEMENT/one"}
    assert all(set(c) >= {"en", "ja", "ex", "exJa", "topic"} and len(c["en"]) <= 80 for c in cards)
    assert any("IPA" in c["exJa"] for c in cards)


def test_tutor_selects_one_and_does_not_repeat_third_turn():
    pytest.importorskip("gradio")
    import app
    pron = [{"category": "TH", "word": "think"}]
    gram = [{"rule": "GRAMMAR", "original": "go", "suggestion": "goes"}]
    selected = app.correction_choice(pron, [], gram, None, "all", set(), [])
    assert selected["kind"] == "pron"
    selected = app.correction_choice(pron, [], gram, None, "all", set(), ["TH", "TH"])
    assert selected["kind"] == "grammar"
    # A turn without a correction breaks the run.
    assert app.correction_choice(pron, [], gram, None, "all", set(), ["TH", None])["kind"] == "pron"
    # Only the pronunciation categories chosen in the settings are corrected.
    assert app.correction_choice(pron, [], gram, None, "all", set(), [], pron_categories=["R_L"])["kind"] == "grammar"
    assert app.correction_sentence({"kind": "pron", "category": "TH", "original": "think", "better": "think"},
                                   "I think so").startswith("Quick note: I think you meant 'think'.")


def test_idle_poll_never_clears_partner_voice(monkeypatch):
    pytest.importorskip("gradio")
    import numpy as np
    import app
    skip = app.gr.skip()
    assert list(app.local_poll()) == [(skip, skip, skip)]
    monkeypatch.setitem(app.active, "mode", "local")
    monkeypatch.setattr(app.rec, "snapshot", lambda: np.zeros(8000, dtype="float32"))
    for _, _, audio in app.local_poll():
        assert audio == skip


def test_cloud_model_is_rejected_before_network():
    c = engine.config()
    c["local_mode"]["ollama_model"] = "example:cloud"
    with pytest.raises(RuntimeError, match="クラウド"):
        engine.ollama_local_preflight(c)
