"""Local UI assets and an explicitly sourced Ollama model guide."""
from html import escape
import gradio as gr

MODEL_CANDIDATES = {
    "llama3.2:3b": {"name": "Llama 3.2 · 3B", "maker": "Meta", "size": "約 2.0 GB", "badge": "最初の候補",
        "description": "英語を含む対話向けの小型モデル。まず英語での会話を試す候補です。日本語は公式の対応言語一覧に含まれません。",
        "tradeoff": "このアプリでの速度・訂正精度は未測定です。複雑な文脈や文法の説明は確認しながら使ってください。",
        "url": "https://ollama.com/library/llama3.2"},
    "qwen2.5:3b": {"name": "Qwen 2.5 · 3B", "maker": "Alibaba / Qwen", "size": "約 1.9 GB", "badge": "日本語も使いたい",
        "description": "英語・日本語を含む多言語に対応。日本語の指示や説明も試したい場合の候補です。このアプリの会話相手は英語で返答します。",
        "tradeoff": "このアプリでの会話品質・訂正精度は未測定です。3B 版には Qwen ライセンスが適用されます。",
        "url": "https://ollama.com/library/qwen2.5"},
    "gemma3:4b": {"name": "Gemma 3 · 4B", "maker": "Google", "size": "約 3.3 GB", "badge": "多言語の別候補",
        "description": "140 以上の言語に対応する Google のモデル。別の会話スタイルを試す候補です。画像にも対応しますが、このアプリは文字だけを渡します。",
        "tradeoff": "Ollama 0.6 以降が必要です。候補の 3B モデルより容量が大きく、実行時のメモリにも余裕が必要です。速度は未測定です。",
        "url": "https://ollama.com/library/gemma3"},
    "llama3.2:1b": {"name": "Llama 3.2 · 1B", "maker": "Meta", "size": "約 1.3 GB", "badge": "小さいモデルから試す",
        "description": "Llama 3.2 のさらに小さい版。3B 版が重い場合に、短い英語のやり取りから試す候補です。",
        "tradeoff": "小型化による品質の差があり得ます。文脈を踏まえた訂正は参考として使い、実際の速度は導入後に確かめてください。",
        "url": "https://ollama.com/library/llama3.2"},
}


def model_choices():
    return [(f"{m['name']} — {m['badge']}", key) for key, m in MODEL_CANDIDATES.items()]


def model_description(model):
    model = (model or "").strip()
    item = MODEL_CANDIDATES.get(model)
    if not item:
        return "### 自分で指定するモデル\n\nOllama に保存済みのローカルモデル名を入力してください。`ollama list` で確認できます。クラウドモデルはこのアプリでは利用できません。", "ollama list"
    text = (f"### {item['name']}\n\n**{item['maker']}** · ダウンロード **{item['size']}** · {item['badge']}\n\n"
            f"{item['description']}\n\n{item['tradeoff']}\n\n"
            f"[公式の説明・ライセンス]({item['url']})\n\n"
            "容量は 2026-09-30 に確認した公式配布の目安です。実行時のメモリ使用量とは異なります。")
    return text, f"ollama pull {model}"


def model_overview():
    cards = "".join(f'<div class="model-mini"><span>{escape(m["badge"])}</span><h3>{escape(m["name"])}</h3>'
                    f'<p>{escape(m["maker"])} · {escape(m["size"])}</p></div>' for m in MODEL_CANDIDATES.values())
    return f'<div class="model-grid">{cards}</div>'


def icon(name, size=24):
    paths = {
        "mic": '<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
        "spark": '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
        "headphones": '<path d="M3 14v-3a9 9 0 0 1 18 0v3"/><rect x="3" y="12" width="4" height="9" rx="2"/><rect x="17" y="12" width="4" height="9" rx="2"/>',
    }
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'


HEADER = f'<header class="app-header"><div class="brand"><span class="brand-icon">{icon("spark",22)}</span>English Talk<span class="brand-sub">英会話の練習室</span></div><span class="local-badge"><i></i>この PC に保存</span></header>'
EMPTY_REPORT = f'<div class="empty-state"><span class="empty-icon">{icon("spark",32)}</span><h2>話したあとに、次の一歩を。</h2><p>会話を終了すると、ここにレポートが届きます。<br>発音・文法・流暢さと、次に練習する重点を振り返れます。</p></div>'
EMPTY_CHAT = f'<div class="chat-empty">{icon("spark",30)}<h3>英語で、話してみよう。</h3><p>会話を始めると、相手が問いかけます。<br>うまく言えなくても、自分の言葉で大丈夫。</p></div>'


def chat_display(log):
    if not log: return EMPTY_CHAT
    rows = []
    for line in log.splitlines():
        who, sep, sentence = line.partition(": ")
        if not sep: who, sentence = "会話", line
        if who == "判定":
            cls = "verdict-pass" if sentence == "合格" else "verdict-fail"
            rows.append(f'<div class="chat-verdict {cls}">{escape(sentence)}</div>')
            continue
        kind = {"自分": "user-turn", "講師": "tutor-turn"}.get(who, "partner-turn")
        rows.append(f'<div class="chat-turn {kind}"><span class="chat-speaker">{escape(who)}</span><div class="chat-bubble">{escape(sentence)}</div></div>')
    return '<div class="chat-messages">' + "".join(rows) + '</div>'
WAVE = "".join(f'<i style="height:{h}px"></i>' for h in [8,14,22,12,30,42,24,16,34,18,10,22,12,8])
RECORDER = f'<div class="recorder-hero"><div class="mic-orbit">{icon("mic",36)}</div><h2>準備ができたら、話してみよう。</h2><p>いつもの会話相手と、いつものアプリで。<br>ここでは、あなたの声を録音します。</p><div class="waveform" aria-hidden="true">{WAVE}</div></div>'


def heading(kicker, title, text):
    return f'<div class="page-heading"><div class="eyebrow">{escape(kicker)}</div><h1>{escape(title)}</h1><p>{escape(text)}</p></div>'


THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="stone",
                       font=["Segoe UI", "Yu Gothic UI", "Meiryo", "sans-serif"],
                       font_mono=["Consolas", "monospace"]).set(
    body_background_fill="#f7f7f3", body_background_fill_dark="#f7f7f3",
    body_text_color="#252b2a", body_text_color_dark="#252b2a",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    input_background_fill="#fafaf8", input_background_fill_dark="#fafaf8",
    button_primary_background_fill="#285c50", button_primary_background_fill_dark="#285c50",
    button_primary_background_fill_hover="#204c42", button_primary_background_fill_hover_dark="#204c42",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff")
