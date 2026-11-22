import argparse
import os
import time
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import lt1_1b_config
from src.engram_moe import compute_ngram_hashes
from src.tokenizer import NanoTokenizer


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.05, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)


def sample_numpy(
    logits: np.ndarray,
    generated_ids: list[int],
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> int:
    logits = logits.astype(np.float64).copy()
    if repetition_penalty > 1.0 and generated_ids:
        for tid in set(generated_ids):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty

    if temperature <= 0.0:
        return int(np.argmax(logits))

    logits = logits / temperature
    logits -= np.max(logits)
    probs = np.exp(logits)
    probs /= np.sum(probs)

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cum_probs = np.cumsum(sorted_probs)
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        probs[sorted_idx[remove]] = 0.0
        probs /= np.sum(probs)

    return int(np.random.choice(len(probs), p=probs))


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LT-1 (1B) ONNX Chat</title>
<style>
  :root { color-scheme: dark; --bg: #0f1117; --card: #181b24; --border: #2a2f3d; --text: #e6e8ef; --muted: #9aa3b2; --accent: #4f8cff; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; background: var(--card); }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header span { font-size: 12px; color: var(--muted); }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; max-width: 860px; width: 100%; margin: 0 auto; }
  .msg { padding: 12px 16px; border-radius: 10px; line-height: 1.5; white-space: pre-wrap; max-width: 85%; font-size: 14px; }
  .user { align-self: flex-end; background: #22355c; border: 1px solid #35528c; }
  .bot { align-self: flex-start; background: var(--card); border: 1px solid var(--border); }
  .meta { font-size: 11px; color: var(--muted); margin-top: 6px; }
  form { border-top: 1px solid var(--border); background: var(--card); padding: 12px 20px; }
  .bar { max-width: 860px; margin: 0 auto; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  input[type=text] { flex: 1; min-width: 240px; padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 14px; }
  input[type=number] { width: 70px; padding: 8px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  button { padding: 10px 18px; border-radius: 8px; border: none; background: var(--accent); color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: wait; }
</style>
</head>
<body>
<header>
  <h1>LT-1 (1B) ONNX Chat</h1>
  <span id="provider">ONNX Runtime</span>
</header>
<div id="chat"></div>
<form id="form">
  <div class="bar">
    <input id="prompt" type="text" placeholder="Ask LT-1 something..." autocomplete="off" required />
    <label>Tokens <input id="max_tokens" type="number" value="64" min="1" max="256" /></label>
    <label>Temp <input id="temp" type="number" value="0.7" min="0" max="2" step="0.1" /></label>
    <button id="send" type="submit">Send</button>
  </div>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');

function addMsg(text, cls, meta = '') {
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = meta;
    div.appendChild(m);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = promptEl.value.trim();
  if (!msg) return;
  addMsg(msg, 'user');
  promptEl.value = '';
  sendBtn.disabled = true;
  const pending = addMsg('Thinking...', 'bot');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        max_new_tokens: parseInt(document.getElementById('max_tokens').value, 10) || 64,
        temperature: parseFloat(document.getElementById('temp').value)
      })
    });
    const data = await res.json();
    pending.textContent = data.reply || '(empty response)';
    const m = document.createElement('div');
    m.className = 'meta';
    m.textContent = `${data.tokens_generated} tokens | ${data.tok_per_sec.toFixed(2)} tok/s`;
    pending.appendChild(m);
  } catch (err) {
    pending.textContent = 'Error: ' + err;
  } finally {
    sendBtn.disabled = false;
    promptEl.focus();
  }
});
fetch('/health').then(r => r.json()).then(d => {
  if (d.provider) document.getElementById('provider').textContent = d.provider;
}).catch(() => {});
</script>
</body>
</html>
"""


def create_app(onnx_path: str, tokenizer_path: str) -> FastAPI:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    tok = NanoTokenizer.load(tokenizer_path)
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(onnx_path, providers=providers or ["CPUExecutionProvider"])
    active_provider = sess.get_providers()[0]
    print(f"ONNX Runtime loaded {onnx_path} with provider: {active_provider}", flush=True)

    engram_buckets = lt1_1b_config().engram_buckets
    app = FastAPI(title="LT-1 ONNX Chat Server")

    @app.get("/health")
    def health():
        return {"status": "ok", "onnx_path": onnx_path, "provider": active_provider}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        ids = [tok.user_id] + tok.encode(req.message) + [tok.eos_id, tok.assistant_id]
        generated: list[int] = []
        t0 = time.perf_counter()
        for _ in range(req.max_new_tokens):
            inp_t = torch.tensor([ids], dtype=torch.long)
            hashes_np = compute_ngram_hashes(inp_t, engram_buckets).numpy()
            logits = sess.run(
                ["logits"],
                {"token_ids": inp_t.numpy(), "ngram_hashes": hashes_np},
            )[0]
            nxt = sample_numpy(
                logits[0, -1, :],
                generated,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            if nxt == tok.eos_id:
                break
            ids.append(nxt)
            generated.append(nxt)
            if len(ids) >= 511:
                break
        dt = max(1e-5, time.perf_counter() - t0)
        return {
            "reply": tok.decode(generated),
            "tokens_generated": len(generated),
            "tok_per_sec": len(generated) / dt,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve LT-1 ONNX model")
    parser.add_argument("--onnx", type=str, default="checkpoints/onnx/lt1_1b_chat.onnx")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer_cosmo.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.onnx, args.tokenizer)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
