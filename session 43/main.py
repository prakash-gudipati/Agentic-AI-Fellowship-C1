"""main.py — the FastAPI app. GET / (form), GET /health, POST /summarize."""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
import summarizer

app = FastAPI(title="Smart Summarizer")


class SummarizeRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.MODEL}


@app.post("/summarize")
def do_summarize(req: SummarizeRequest) -> dict:
    return summarizer.summarize(req.text)


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Summarizer</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto;
           padding: 0 16px; background:#0d0d2b; color:#e8e8f0; }
    h1 { color:#b44fff; }
    textarea { width:100%; height:180px; padding:12px; border-radius:8px;
               border:1px solid #2a2a4a; background:#08081a; color:#e8e8f0; font-size:15px; }
    button { margin-top:12px; padding:10px 20px; border:0; border-radius:8px;
             background:#b44fff; color:#fff; font-size:16px; cursor:pointer; }
    .out { margin-top:24px; padding:16px; border-radius:8px; background:#08081a;
           border:1px solid #2a2a4a; display:none; }
    .tldr { font-size:17px; margin-bottom:12px; }
    li { margin:6px 0; }
  </style>
</head>
<body>
  <h1>Smart Summarizer</h1>
  <p>Paste any text. Get a TL;DR and 3 key points.</p>
  <textarea id="text" placeholder="Paste an article, email, or notes..."></textarea>
  <button onclick="run()">Summarize</button>
  <div class="out" id="out">
    <div class="tldr" id="tldr"></div>
    <ul id="points"></ul>
  </div>
  <script>
    async function run() {
      const text = document.getElementById('text').value;
      const out = document.getElementById('out');
      const btn = document.querySelector('button');
      btn.textContent = 'Summarizing...'; btn.disabled = true;
      try {
        const r = await fetch('/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await r.json();
        document.getElementById('tldr').textContent = data.tl_dr;
        const ul = document.getElementById('points'); ul.innerHTML = '';
        (data.key_points || []).forEach(p => {
          if (!p) return;
          const li = document.createElement('li'); li.textContent = p; ul.appendChild(li);
        });
        out.style.display = 'block';
      } finally {
        btn.textContent = 'Summarize'; btn.disabled = false;
      }
    }
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE
