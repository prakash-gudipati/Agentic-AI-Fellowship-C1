import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ─── PDF 1 — Intro to RAG (research-style) ──────────────────────────────────

INTRO_RAG_TEXT = [
    ("Introduction to Retrieval-Augmented Generation",
     "title"),
    ("Abstract", "h2"),
    ("Retrieval-Augmented Generation (RAG) is a technique that lets a Large "
     "Language Model answer questions using a corpus of documents it was never "
     "trained on. The technique combines a retrieval step over a knowledge base "
     "with a generation step performed by the language model. RAG was first "
     "introduced by Lewis et al. in 2020 as a way to ground language model "
     "outputs in verifiable sources.", "p"),
    ("1. The Problem RAG Solves", "h2"),
    ("Large Language Models have two well-known failure modes when asked about "
     "topics outside their training distribution. First, they hallucinate — "
     "producing confident text that has no basis in fact. Second, they are "
     "limited by their knowledge cutoff date and cannot answer questions about "
     "events, products, or documents that did not exist at training time. RAG "
     "addresses both problems by injecting relevant source material into the "
     "model's context at query time.", "p"),
    ("2. The Five Stages of a RAG Pipeline", "h2"),
    ("A standard RAG pipeline consists of five sequential stages: ingest, chunk, "
     "embed, retrieve, and generate. The first three stages run at index time, "
     "typically once when the corpus is updated. The last two stages run at "
     "query time, once per user question. The separation between index-time and "
     "query-time work is what allows RAG systems to scale to millions of "
     "documents while still answering questions in milliseconds.", "p"),
    ("3. Chunking and Why It Matters", "h2"),
    ("A chunk is a small contiguous piece of a source document. Chunks exist "
     "for two reasons. First, language model context windows are finite — even "
     "a 200,000-token window cannot hold a million-document corpus. Second, "
     "retrieval is more precise on small chunks than on full documents, because "
     "an embedding of a 50-page document averages out the meaning of every "
     "page. A typical chunk size is 500 to 1500 characters with 10-20 percent "
     "overlap between neighbours, but optimal sizes are domain-specific.", "p"),
    ("4. Grounding the Generation Step", "h2"),
    ("After retrieval returns the top-K most-relevant chunks, the generation "
     "step composes a prompt of the form: SYSTEM = answer using only the "
     "context below. CONTEXT = retrieved chunks. QUESTION = the user query. "
     "This prompt structure, combined with explicit instructions to refuse to "
     "answer when the context is insufficient, is what reduces hallucination. "
     "Grounding is not automatic — it must be requested in the prompt.", "p"),
    ("5. Failure Modes", "h2"),
    ("RAG systems still fail in three predictable ways. Out-of-corpus "
     "questions return no relevant chunks and require the model to refuse "
     "honestly. Ambiguous questions retrieve the wrong chunks. And retrieval "
     "misses occur when the embedding model fails to map a paraphrased query "
     "to the correct chunk. Sessions 21 through 28 of this fellowship address "
     "each of these failure modes in turn.", "p"),
    ("References", "h2"),
    ("Lewis, P. et al. Retrieval-Augmented Generation for Knowledge-Intensive "
     "NLP Tasks. NeurIPS 2020.", "p"),
]


# ─── PDF 2 — Product Manual ─────────────────────────────────────────────────

PRODUCT_MANUAL_TEXT = [
    ("WidgetMax 3000 — User Manual", "title"),
    ("Page 1 of 2  |  Document Ref: WM3K-UM-v2.4  |  © Acme Corp", "footer"),
    ("Chapter 1 — Installation", "h2"),
    ("Thank you for purchasing the WidgetMax 3000. To install the device, "
     "first remove all packing material and verify the contents of the box: "
     "one main unit, one power adapter, two mounting screws, and one quick-"
     "start card. Place the main unit on a flat, dry surface within two metres "
     "of a power outlet. Connect the power adapter to the round port labelled "
     "DC-IN on the back of the unit. Press the power button for three seconds "
     "until the front LED turns solid green.", "p"),
    ("Chapter 2 — First-Time Setup", "h2"),
    ("On first power-on, the WidgetMax 3000 enters setup mode. The front LED "
     "blinks blue for thirty seconds. During this window, open the WidgetMax "
     "companion app on your phone and tap Add Device. The app will detect the "
     "unit automatically. Enter your home Wi-Fi password when prompted. The "
     "LED turns solid green when the unit has joined the network. If the LED "
     "remains blinking after one minute, hold the reset pinhole on the back of "
     "the unit for ten seconds and start over.", "p"),
    ("Page 2 of 2  |  WidgetMax 3000 User Manual", "footer"),
    ("Chapter 3 — Daily Use", "h2"),
    ("Once installed, the WidgetMax 3000 runs in the background. The front "
     "LED gives you status at a glance. Solid green means everything is "
     "working. Slow amber pulse means a firmware update is in progress — do "
     "not unplug the unit. Solid red means the unit has lost contact with the "
     "WidgetMax cloud — check your internet connection. The unit consumes "
     "about 4 watts during normal operation and up to 12 watts during firmware "
     "updates.", "p"),
    ("Chapter 4 — Troubleshooting", "h2"),
    ("If the unit will not power on, verify the power adapter is fully seated "
     "and the outlet is live. If the LED is solid red for more than ten "
     "minutes, the unit cannot reach the cloud — restart your router and "
     "wait two minutes. If the companion app cannot find the device, ensure "
     "your phone and the WidgetMax 3000 are on the same Wi-Fi network. For "
     "all other issues, contact support@acme-widgetmax.example with the "
     "serial number printed on the bottom of the unit.", "p"),
    ("Warranty: 2 years from date of purchase. Support hours: Mon-Fri 09:00-"
     "18:00 IST and 09:00-17:00 PT.", "footer"),
]


# ─── HTML — Sample article with messy markup ────────────────────────────────

SAMPLE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <title>Why RAG Beats Fine-Tuning for Most Production Use-Cases | Acme Engineering Blog</title>
  <meta charset="utf-8">
</head>
<body>

<header class="site-header">
  <nav class="primary-nav">
    <a href="/">Home</a>
    <a href="/blog">Blog</a>
    <a href="/products">Products</a>
    <a href="/careers">Careers</a>
    <a href="/contact">Contact Us</a>
  </nav>
  <div class="search-bar">
    <input type="search" placeholder="Search the blog...">
    <button>Go</button>
  </div>
  <div class="cookie-banner">
    We use cookies to improve your experience. By using this site you agree to our cookie policy.
    <button>Accept</button>
    <button>Decline</button>
  </div>
</header>

<aside class="sidebar">
  <h3>Recent posts</h3>
  <ul>
    <li><a href="/blog/post-1">Building a CLI in Python</a></li>
    <li><a href="/blog/post-2">When to use Anthropic vs OpenAI</a></li>
    <li><a href="/blog/post-3">Embeddings explained for product managers</a></li>
  </ul>
  <div class="ad-slot">[Sponsored — your ad here]</div>
</aside>

<main>
<article>
  <h1>Why RAG Beats Fine-Tuning for Most Production Use-Cases</h1>
  <p class="byline">By the Acme Engineering team — published 12 March 2026 — 7 min read</p>

  <p>Every team that builds with Large Language Models eventually asks the same question: should we fine-tune the model on our private data, or should we use Retrieval-Augmented Generation? In our experience running production AI workloads for over forty enterprise customers, the answer is RAG in nine out of ten cases. Here is why.</p>

  <h2>RAG separates knowledge from reasoning</h2>
  <p>A fine-tuned model bakes your private knowledge into its weights. To update a single document you must re-run the entire training job, which costs hundreds of dollars and takes hours. With RAG, your knowledge lives in a separate index — change a document, re-embed only that chunk, and the system is updated in seconds. Reasoning ability comes from the base model. Knowledge comes from your index. Keeping them separate is the key insight.</p>

  <h2>RAG is auditable. Fine-tuning is a black box.</h2>
  <p>When a RAG system answers a question, you can show the user exactly which chunks it used to compose the answer. This matters in regulated industries — healthcare, legal, financial services — where every claim must be traceable to a source. A fine-tuned model produces a confident sentence with no provenance. You cannot tell which training example contributed to which output token.</p>

  <h2>RAG fails gracefully on out-of-domain questions</h2>
  <p>If a user asks a RAG system a question whose answer is not in the index, retrieval returns low-similarity chunks and the prompt template instructs the model to refuse. A fine-tuned model has no such circuit breaker — it will hallucinate confidently because nothing in its prompt told it to stop. Refusal is a feature you can engineer in RAG and almost cannot engineer in fine-tuning.</p>

  <h2>When fine-tuning still wins</h2>
  <p>Fine-tuning beats RAG when the desired output is a stylistic transformation rather than a knowledge retrieval. If you want the model to respond in your brand voice, or to consistently structure its output as a specific schema, fine-tuning teaches that pattern more reliably than prompting. But these cases are the exception, not the rule.</p>

  <p>For the rest of this series, we will build a production-grade RAG system from scratch, with no framework, in pure Python.</p>
</article>
</main>

<footer class="site-footer">
  <p>© 2026 Acme Corp. All rights reserved.</p>
  <p>Privacy Policy | Terms of Service | Do Not Sell My Personal Information</p>
  <nav class="footer-nav">
    <a href="/about">About</a>
    <a href="/press">Press</a>
    <a href="/investors">Investors</a>
    <a href="/security">Security</a>
  </nav>
</footer>

<script>
  // Analytics tracking, ignored at this point
  console.log("page view tracked");
</script>

</body>
</html>
"""


def build_pdf(filename: str, blocks: list) -> None:
    """Render a list of (text, style_tag) blocks into a 2-page PDF."""
    path = os.path.join(DATA_DIR, filename)
    doc = SimpleDocTemplate(
        path, pagesize=LETTER,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
    )

    base = getSampleStyleSheet()
    title = ParagraphStyle("title",  parent=base["Title"],   alignment=TA_LEFT, fontSize=18, spaceAfter=14)
    h2    = ParagraphStyle("h2",     parent=base["Heading2"],alignment=TA_LEFT, fontSize=13, spaceAfter=8)
    body  = ParagraphStyle("body",   parent=base["BodyText"],alignment=TA_LEFT, fontSize=11, leading=15, spaceAfter=10)
    footer= ParagraphStyle("footer", parent=base["BodyText"],alignment=TA_LEFT, fontSize=8,  textColor="#888888", spaceAfter=10)

    flow = []
    for text, tag in blocks:
        if tag == "title":
            flow.append(Paragraph(text, title))
        elif tag == "h2":
            flow.append(Paragraph(text, h2))
        elif tag == "footer":
            flow.append(Paragraph(text, footer))
        else:
            flow.append(Paragraph(text, body))
        flow.append(Spacer(1, 4))
    doc.build(flow)
    print(f"[generate_corpus] wrote {path}")


def build_html(filename: str, content: str) -> None:
    """Write the sample HTML article."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[generate_corpus] wrote {path}")


def main() -> None:
    print("[generate_corpus] building sample messy corpus in", DATA_DIR)
    build_pdf("intro_to_rag.pdf", INTRO_RAG_TEXT)
    build_pdf("product_manual.pdf", PRODUCT_MANUAL_TEXT)
    build_html("sample_article.html", SAMPLE_HTML)
    print("[generate_corpus] done. 3 source documents ready.")


if __name__ == "__main__":
    main()
