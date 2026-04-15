# AI After-Sales Assistant (Ahooga)

An AI-powered after-sales support system built during my internship at Ahooga Bike.

This project is designed to provide accurate, context-aware answers to customer support queries by leveraging Retrieval-Augmented Generation (RAG) instead of traditional chatbot approaches.

---

## 🚀 Overview

The system transforms scattered technical documentation, support tickets, and business data into a structured AI-powered assistant.

Instead of generating answers blindly, the system:
1. Retrieves relevant technical information
2. Uses AI to generate grounded responses

---

## 🧠 Architecture

Frontend (React)
↓
FastAPI (Backend Logic Layer)
↓
Azure OpenAI (LLM)
↓
Azure AI Search (Vector Database)
↑
Document + Ticket Ingestion Pipeline
↑
Odoo API (Business Data)

---

## ⚙️ Tech Stack

### Backend
- FastAPI (Python)
- Uvicorn

### AI / ML
- Azure OpenAI (GPT + Embeddings)
- Retrieval-Augmented Generation (RAG)

### Data & Search
- Azure AI Search (Vector DB)

### Data Processing
- PyMuPDF (PDF extraction)
- Tesseract OCR (image text extraction)

### Integration
- Odoo API (ERP system)

### Frontend
- React (Vite)

### DevOps
- Docker
- Azure Container Registry
- ngrok (testing)

---

## 🔥 Key Features

- AI-powered support assistant
- Semantic search over technical documents
- Ticket ingestion and structuring
- Modular architecture (not a simple chatbot)
- Context-aware responses (no hallucination approach)

---

## ⚠️ Note

Due to company policy, a live public demo is not available.

---

## 👨‍💻 Author

Paschal Ifewulu  
Computer Science (AI) Student  
