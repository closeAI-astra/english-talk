"""Read-only local readiness check. Does not download or open a microphone."""
from __future__ import annotations

import importlib.util
import shutil
import urllib.request

from pathlib import Path

from engine import ROOT, config


def main():
    for module in ("gradio", "sounddevice", "soundfile", "silero_vad", "faster_whisper",
                   "torch", "transformers", "phonemizer", "espeakng_loader",
                   "language_tool_python", "wordfreq"):
        print(f"{module}: {'OK' if importlib.util.find_spec(module) else '未導入'}")
    c = config()
    print("Java:", shutil.which("java") or "未検出")
    print("Piper:", shutil.which(c["tts"]["piper_exe"]) or "未検出")
    voice = c["tts"]["piper_voice"]
    print("Piper 音声モデル:", (voice + (" (OK)" if voice and (ROOT / voice).is_file() or Path(voice).is_file() else " (ファイルなし)")) if voice else "未設定")
    print("DeepFilterNet（任意）:", "OK" if importlib.util.find_spec("df") else "未導入（ノイズ除去を使う場合のみ必要）")
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            print("Ollama:", "起動中" if r.status == 200 else f"HTTP {r.status}")
    except Exception:
        print("Ollama: 未起動")
    print("Ollama モデル:", c["local_mode"]["ollama_model"] or "未設定")
    if importlib.util.find_spec("sounddevice"):
        import sounddevice as sd
        try:
            print("入力マイク:", [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0])
        except Exception as e:
            print("入力マイク: 確認できません -", e)


if __name__ == "__main__":
    main()
