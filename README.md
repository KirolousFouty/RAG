# iPhone User Guide — RAG chatbot

A retrieval-augmented chatbot to ask questions about the Apple iPhone User Guide
(iOS 7.1). Answers are grounded strictly in the PDF and cite the page they come from;
if something isn't in the guide, the bot says so instead of guessing.

Work in progress. Design notes will land under `docs/` as I make the decisions.

## Planned stack

- LangChain / LangGraph for orchestration
- Chainlit for the chat UI
- Pinecone (serverless free tier) for the vector store
- bge-small embeddings, Claude as the chat model
- Single Docker container to run it

## Layout

- `src/` — ingestion + RAG pipeline
- `eval/` — retrieval experiments
- `docs/` — design notes
