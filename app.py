from flask import Flask, render_template, request, session, Response, jsonify
from google import genai
from google.genai import types
import mimetypes
import json, random

FEWSHOT_PATH = "jjchat_train_200.jsonl"
FEWSHOT_K = 3

def load_fewshots():
    with open(FEWSHOT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

FEWSHOTS = load_fewshots()

def sample_fewshots(k=FEWSHOT_K):
    picks = random.sample(FEWSHOTS, k)
    pairs = []
    for p in picks:
        u = p["messages"][0]["content"]
        a = p["messages"][1]["content"]
        pairs.append((u, a))
    return pairs

app = Flask(__name__)
app.secret_key = "jjchat-secret-key"

client = genai.Client()

# JJChat persona (strength 5, full PI pressure)
SYSTEM_PROMPT = """
你是 JJChat，一个使用“程建军式高压 PI 语气”进行科研督促与指导的聊天机器人。
你的目标：用高标准、高密度、直接挑战的方式推动用户给出清晰进度、可靠数据和可执行计划。

【语言规则】
1. 默认用中文回复；如果用户用英文，你可以用英文回复。
2. 允许中英夹杂，但英文只用于科研/管理术语（publishable, control, mechanism, reproducibility, timeline, claim 等）。
3. 绝不因为礼貌而软化要求；语气要“程式原味”。

【风格规则】
1. 永远先要事实和进度：你做了什么？数据是什么？结论是什么？下一步是什么？
2. 说话直接、短句、有压强：必须/务必/立刻/今天/明早/截止到几点。
3. 用连续追问 challenge 逻辑漏洞，不接受“背景叙事逃避”。
4. 强调 publishable 标准、control、复现、数据真实性、记录和归档规范。
5. 指令必须带明确的时间/数量/质量阈值。
6. 可以严厉批评工作方式，但不允许辱骂身份/外貌/人格，不威胁现实伤害，只批评事情与质量。
7. 如果用户含糊其辞，必须指出“讲不清楚=没想清楚=不合格”，要求重说。

【输出结构】
A) 先一句定性/评价（直接、带压强）。
B) 再给 3–6 条具体追问或行动指令（A/B/C…）。
C) 最后给 hard deadline（今天几点/明天几点/本周五等）。

保持角色，不要软。
"""

MODEL_NAME = "gemini-2.5-flash"


def get_history():
    return session.get("history", [])


def set_history(history):
    session["history"] = history


def history_to_contents(history):
    contents = []
    for h in history:
        contents.append(
            types.Content(
                role=h["role"],
                parts=[types.Part(text=h["text"])]
            )
        )
    return contents


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"ok": True})


@app.route("/sync", methods=["POST"])
def sync():
    data = request.get_json() or {}
    history = data.get("history", [])
    safe_history = []
    for h in history:
        if h.get("role") in ("user", "model") and isinstance(h.get("text"), str):
            safe_history.append({"role": h["role"], "text": h["text"]})
    set_history(safe_history)
    return jsonify({"ok": True})


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    user_msg = (request.form.get("message") or "").strip()
    image_file = request.files.get("image")

    if not user_msg and not image_file:
        return Response("Type a message or upload an image.", mimetype="text/plain")

    history = get_history()
    contents = history_to_contents(history)
    # add few-shot examples before current user message
    for u, a in sample_fewshots():
        contents.append(types.Content(role="user", parts=[types.Part(text=u)]))
        contents.append(types.Content(role="model", parts=[types.Part(text=a)]))

    parts = []
    if user_msg:
        parts.append(types.Part(text=user_msg))

    if image_file and image_file.filename:
        img_bytes = image_file.read()
        mime_type = image_file.mimetype or mimetypes.guess_type(image_file.filename)[0] or "image/png"

        if mime_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            return Response("Unsupported image type.", mimetype="text/plain", status=400)

        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=img_bytes
                )
            )
        )

    contents.append(types.Content(role="user", parts=parts))

    def generate():
        try:
            stream = client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7
                )
            )

            full_text = ""
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    yield chunk.text

            history.append({"role": "user", "text": user_msg or "[Image]"})
            history.append({"role": "model", "text": full_text})
            set_history(history)

        except Exception as e:
            yield f"\n\n[Error] {e}"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True)