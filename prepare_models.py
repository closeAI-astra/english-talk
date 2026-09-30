"""Download model weights once. No recordings or transcripts are uploaded.

Run explicitly while connected; app.py uses cached files only thereafter.
"""
from __future__ import annotations

from engine import DATA, config, model_dir


def main():
    c = config()
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download, hf_hub_download
    models = list(dict.fromkeys([c["asr"]["model"], c["tutor"]["asr_model"]]))
    for name in models:
        print("ASR:", name, flush=True)
        WhisperModel(name, device="cpu", compute_type="int8", local_files_only=False)
    for key, kind in ((c["phoneme"]["model"], "phoneme"),
                      (c["intent"]["lm"], "intent")):
        print("Model:", key, flush=True)
        snapshot_download(key, local_dir=str(model_dir(kind, key)),
                          allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"])
    voice_dir = DATA / "voices"
    for filename in ("en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"):
        hf_hub_download("rhasspy/piper-voices",
                        f"en/en_US/lessac/medium/{filename}", repo_type="model",
                        local_dir=str(voice_dir))
    voice = voice_dir / "en" / "en_US" / "lessac" / "medium" / "en_US-lessac-medium.onnx"
    if not c["tts"]["piper_voice"]:
        from engine import save_config
        c["tts"]["piper_voice"] = voice.relative_to(DATA.parent).as_posix()
        save_config(c)
    print("モデルを保存しました。ローカル会話用の Ollama モデルは別途設定してください。")


if __name__ == "__main__":
    main()
