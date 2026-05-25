# Deliverable 3: Demo + Web Interface (20%)

> **Deadline**: Week 12 — Working Demo + Presentation + Supplementary Materials

## Overview

A premium web chat interface themed as "Better Call Saul AI" — where users interact with an AI legal assistant for UAE law questions. Every answer is grounded in specific legal articles.

---

## Features

### Chat Interface
- Real-time Q&A with the RAG pipeline
- Saul Goodman persona for engaging responses
- Message history with styled bubbles (user = blue, AI = dark gold)
- Typing indicator animation while processing

### Sources Panel
- Collapsible sidebar showing retrieved legal articles
- Source relevance scores displayed as percentage
- Click to expand full article text

### Confidence Score
- Visual confidence meter (0–100%)
- Color coded: green (>70%), amber (40–70%), red (<40%)

### Hallucination Detection
- Pulsing red warning when potential hallucinations detected
- Lists specific flagged claims

### Suggestion Chips
- 5 starter questions for new users
- One-click to ask common UAE law questions

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask (Python) |
| Frontend | HTML5 + Vanilla CSS + Vanilla JS |
| Fonts | Google Fonts (Inter, Playfair Display) |
| Theme | Dark mode with glassmorphism |
| Colors | Gold (#d4a843), Dark (#0a0a0f) |

---

## How to Run

```bash
# 1. First, ingest documents into the vector store
python deliverable_1/scripts/ingest_documents.py

# 2. Launch the web interface
python deliverable_3/app/app.py

# 3. Open in browser
# http://localhost:5000
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/api/query` | POST | Submit a query (`{"query": "..."}`) |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Vector store statistics |

### Query Response Format
```json
{
    "answer": "Well, counselor, let me break this down...",
    "sources": [
        {
            "text": "Article 37: ...",
            "law_name": "UAE Penal Code",
            "article_number": "37",
            "score": 0.8542
        }
    ],
    "confidence": 0.85,
    "hallucination_flags": [],
    "timing": {
        "retrieval": 0.234,
        "generation": 1.456,
        "total": 1.690
    }
}
```

---

## Design Choices

- **Dark theme**: Professional, premium feel appropriate for legal content
- **Gold accents**: Saul Goodman's signature color — immediately recognizable
- **Glassmorphism**: Modern aesthetic with backdrop-filter blur effects
- **Smooth animations**: Message slide-ins, typing dots bounce, confidence fill
- **Responsive layout**: Works on desktop and mobile
- **Legal disclaimer**: Always visible to set expectations
