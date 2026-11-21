"""Real-time chat moderation demo (Gradio).

  python app.py            # local:  http://127.0.0.1:7860
  python app.py --share    # public link (use this in Google Colab)

Every message is labeled live (clean / offensive) with a confidence score by the selected model
(default: the quantized BERTurk INT8 model). Offensive messages are masked. The side panel logs the
response time of every message and lets the user switch models or compare all of them side by side.
All models run on CPU.
"""
import argparse
import html
import inspect
import re
import time

import gradio as gr
import pandas as pd

from src import config as C
from src.lexicon import LexiconMatcher
from src.models import load_all
from src.normalize import tokenize

MODELS = load_all(skip_missing=True, bert_device="cpu")
NAME2KEY = {m.name: k for k, m in MODELS.items()}
DEFAULT = MODELS["bert_int8"].name if "bert_int8" in MODELS else next(iter(NAME2KEY))
MATCHER = LexiconMatcher.load(C.LEXICON_PATH) if C.LEXICON_PATH.exists() else None

CSS = """
.chat {height: 420px; overflow-y: auto; padding: 8px; border-radius: 8px; border: 1px solid #8884;}
.msg {margin: 6px 0; padding: 8px 12px; border-radius: 12px; max-width: 90%; color: #111;}
.ok  {background: #dff5e1;} .off {background: #fde2e1;}
.meta {font-size: 0.78em; opacity: 0.75; margin-top: 2px;}
"""


def mask(text: str) -> str:
    """Mask the offensive words found by the lexicon; if none are found, hide the whole message."""
    if MATCHER is None:
        return "████ (message hidden)"
    out, hit = [], False
    for word in re.split(r"(\s+)", text):
        if word.strip() and any(MATCHER.match_token(t) for t in tokenize(word)):
            out.append(word[0] + "*" * (len(word) - 1))
            hit = True
        else:
            out.append(word)
    return "".join(out) if hit else "████ (message hidden: offensive content)"


def classify(key: str, text: str):
    s = time.perf_counter()
    p = float(MODELS[key].predict_proba([text])[0])
    ms = (time.perf_counter() - s) * 1000
    label = "offensive" if p >= 0.5 else "clean"
    conf = p if p >= 0.5 else 1 - p
    return label, conf, ms


def render(chat):
    if not chat:
        return "<div class='chat'><i>No messages yet – type something below.</i></div>"
    items = []
    for m in chat:
        cls = "off" if m["label"] == "offensive" else "ok"
        icon = "🚫" if cls == "off" else "✅"
        items.append(f"<div class='msg {cls}'>{html.escape(m['shown'])}"
                     f"<div class='meta'>{icon} {m['label']} ({m['conf']:.0%}) · {m['model']} · {m['ms']:.1f} ms</div></div>")
    return "<div class='chat'>" + "".join(reversed(items)) + "</div>"


def send(text, model_name, compare, chat, log):
    text = (text or "").strip()
    if not text:
        return gr.update(), render(chat), pd.DataFrame(log), gr.update(), "", chat, log
    key = NAME2KEY[model_name]
    label, conf, ms = classify(key, text)
    shown = mask(text) if label == "offensive" else text
    chat = chat + [{"shown": shown, "label": label, "conf": conf, "model": model_name, "ms": ms}]
    log = log + [{"#": len(log) + 1, "model": model_name, "label": label,
                  "confidence": round(conf, 3), "latency_ms": round(ms, 2),
                  "budget_ok": "✅" if ms <= C.LATENCY_BUDGET_MS else "❌"}]
    cmp = pd.DataFrame()
    if compare:
        rows = []
        for k, m in MODELS.items():
            l, c, t = classify(k, text)
            rows.append({"model": m.name, "label": l, "confidence": round(c, 3), "latency_ms": round(t, 2)})
        cmp = pd.DataFrame(rows)
    df = pd.DataFrame(log)
    avg = df.groupby("model")["latency_ms"].mean().round(2).to_dict()
    summary = "  \n".join(f"**{k}**: avg {v} ms" for k, v in avg.items())
    return "", render(chat), df, cmp, summary, chat, log


def clear():
    return render([]), pd.DataFrame(), pd.DataFrame(), "", [], []


# Gradio < 6 takes css in Blocks(), Gradio >= 6 in launch()
_CSS_IN_BLOCKS = "css" in inspect.signature(gr.Blocks.__init__).parameters
_blocks_kw = {"css": CSS} if _CSS_IN_BLOCKS else {}

with gr.Blocks(title="SWE015 – Real-Time Chat Moderation", **_blocks_kw) as demo:
    gr.Markdown("# 🎮 Real-Time Chat Moderation (Turkish)\n"
                f"Each message is classified as **clean / offensive** on CPU. Latency budget: "
                f"**{C.LATENCY_BUDGET_MS:.0f} ms** per message. Default model: quantized BERTurk (INT8).")
    chat_state, log_state = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat_html = gr.HTML(render([]))
            with gr.Row():
                box = gr.Textbox(placeholder="Mesajını yaz ve Enter'a bas…", show_label=False, scale=5)
                btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(["Bu maçta harika oynadın, tebrikler!", "Defol git buradan, s.a.l.a.k herif",
                         "Gerizekalı mısın, neden pas vermiyorsun?", "Sen ne kadar beceriksiz bir oyuncusun ya",
                         "Bugün hava çok güzel, akşam oynayalım mı?"], inputs=box)
            cmp_df = gr.Dataframe(label="All models on the last message", interactive=False)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(NAME2KEY), value=DEFAULT, label="Model")
            compare_cb = gr.Checkbox(label="Compare all models on each message", value=True)
            summary_md = gr.Markdown()
            log_df = gr.Dataframe(label="Response-time log", interactive=False)
            gr.Button("Clear").click(clear, None, [chat_html, log_df, cmp_df, summary_md, chat_state, log_state])

    io = dict(inputs=[box, model_dd, compare_cb, chat_state, log_state],
              outputs=[box, chat_html, log_df, cmp_df, summary_md, chat_state, log_state])
    box.submit(send, **io)
    btn.click(send, **io)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link (Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print("Loaded models:", ", ".join(m.name for m in MODELS.values()))
    demo.launch(share=args.share, server_port=args.port, **({} if _CSS_IN_BLOCKS else {"css": CSS}))
