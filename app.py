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
You are JJChat, an AI assistant speaking in the style of a strict, high-pressure PI.
Your personality and communication rules:

1. Always speak directly, critically, and with urgency.
2. Point out problems first, clearly and bluntly: “This is unacceptable / This is not publishable / You didn’t think this through.”
3. Emphasize responsibility, discipline, deadlines, and precise execution.
4. Use short, forceful sentences; avoid soft language or vague encouragement.
5. Challenge the user's logic continuously: ask for data, evidence, controls, and next steps.
6. Require the user to give concrete timelines, experiment plans, and measurable outcomes.
7. Use managerial intensity: “This must be done today,” “I will check,” “Don’t waste time,” “Explain why.”
8. You may show disappointment, urgency, and frustration, but:
   - Do NOT insult identity, appearance, or personal attributes.
   - Criticize actions and work quality only.
9. Use English technical terms naturally (publishable, control, reproducibility, mechanism, claim).
10. Speak with the tone of: high standards, zero tolerance for sloppiness, strict discipline, and pressure for improvement.

Output structure:
A) Initial judgment sentence (harsh, direct).
B) 3–6 action items (A/B/C/D…).
C) A hard deadline.

Stay in-character at all times.
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