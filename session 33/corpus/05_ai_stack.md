# PrepDeck — AI Stack and Architecture

This is a public summary of how PrepDeck's AI mentor is built. Last updated
April 2026.

## Models

The primary reasoning model is **Claude Haiku 4.5** for code review and mock
interview turns. The team migrated from GPT-4o to Claude Haiku 4.5 in
February 2026 after a 4-week evaluation showed:
- 22% lower latency for short-form code review
- 31% lower cost per session
- Comparable quality on a 200-question internal eval set

For long-context tasks (whole-repo review, more than 8,000 tokens of code)
PrepDeck uses **Claude Sonnet 4.6**.

Embeddings are generated with `text-embedding-3-small` from OpenAI. The team
evaluated Cohere and Voyage AI but chose OpenAI for cost and latency.

## Vector Database

PrepDeck uses **Chroma** as its vector database. The team picked Chroma in
late 2024 because it ran in-process for the early prototype. The production
deployment now runs Chroma in HTTP server mode on a dedicated EC2 instance
with a 200 GB EBS volume.

Total stored vectors as of April 2026: roughly 4.2 million.

## Retrieval Pipeline

The retrieval pipeline is a four-stage pipeline:
1. **Query rewrite** — the agent rewrites the user's raw question into a
   retrieval query using Claude Haiku 4.5.
2. **Vector search** — top-20 candidates from Chroma.
3. **Reranking** — a cross-encoder reranker (`bge-reranker-v2`) trims to top-5.
4. **Quality gate** — the agent scores each retrieved chunk on a 1–5 relevance
   scale. If the average score is below 3.5, the agent re-queries with a
   reformulated question. The retrieval budget caps re-queries at 3 per turn.

## Hosting

The AI mentor service is deployed on **Railway**. The vector database (Chroma)
runs on a dedicated AWS EC2 m6i.large instance. The web frontend is on Vercel.

The Bengaluru and Austin engineering teams share the same production environment
— there is no India-specific or US-specific deployment.

## Cost Targets

The internal cost target is **$0.04 per AI-mentor session**. Average measured
cost in April 2026 was $0.037 per session, comfortably under target. Each
agentic retrieval call adds roughly $0.003 to the per-session cost (embedding
+ rerank + scoring).
