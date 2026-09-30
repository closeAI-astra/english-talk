"""Evaluate user-recorded minimal pairs, without inventing test audio.

Filename: TH__think__wrong__01.wav or TH__think__correct__01.wav.
The target word is what the learner intended to say, even in a wrong sample.
"""
from __future__ import annotations

import csv
from pathlib import Path

from engine import RATE, config, pronunciation, read_wav, release_phoneme

ROOT = Path(__file__).resolve().parent
CATEGORIES = {"R_L", "TH", "V_B", "VOWEL", "EPENTHESIS", "F_H"}


def main():
    folder = ROOT / "tests" / "audio"
    files = sorted(folder.glob("*.wav"))
    if not files:
        raise SystemExit("tests/audio/ に本人の録音がありません。README を参照してください。")
    rows = []
    try:
        for path in files:
            parts = path.stem.split("__")
            if len(parts) != 4 or parts[0] not in CATEGORIES or parts[2] not in ("correct", "wrong"):
                print("名前を確認してください:", path.name)
                continue
            category, target, truth, _ = parts
            y = read_wav(path)
            word = {"w": target, "start": 0, "end": len(y) / RATE}
            utterance = [{"id": 0, "start": 0, "end": len(y) / RATE,
                          "text": target, "words": [word]}]
            try:
                errors = pronunciation(utterance, y, config())
                detected = any(e["category"] == category for e in errors)
                held = word.get("conf", 0) < config()["phoneme"]["min_confidence"]
                error = ""
            except Exception as exc:
                detected, held, error = False, True, str(exc)
            rows.append({"file": path.name, "category": category, "truth": truth,
                         "detected": detected, "held": held, "error": error})
    finally:
        release_phoneme()
    out = ROOT / "tests" / "audio_results.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file","category","truth","detected","held","error"])
        writer.writeheader(); writer.writerows(rows)
    eligible = [r for r in rows if not r["held"]]
    wrong = [r for r in eligible if r["truth"] == "wrong"]
    correct = [r for r in eligible if r["truth"] == "correct"]
    print("録音数", len(rows), "判定対象", len(eligible), "判定保留", len(rows)-len(eligible))
    print("誤発音の検出率", sum(r["detected"] for r in wrong)/len(wrong) if wrong else "判断不能")
    print("正しい発音の誤検出率", sum(r["detected"] for r in correct)/len(correct) if correct else "判断不能")
    print("内訳:", out)


if __name__ == "__main__":
    main()
