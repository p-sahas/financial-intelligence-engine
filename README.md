# Financial Intelligence Engine

> **"The Intern" vs. "The Librarian"**
> An advanced AI system for analyzing financial reports, comparing Fine-Tuning strategies against Retrieval-Augmented Generation (RAG).

##  Overview

The **Financial Intelligence Engine** is a comprehensive GenAI project designed to ingest, process, and analyze complex financial documents (specifically the Uber 2024 Annual Report). The project implements and compares two distinct architectural approaches to answer financial queries:

1.  **The Librarian (RAG)**: A Retrieval-Augmented Generation system using Weaviate (Hybrid Search) and Cross-Encoder reranking to fetch precise context before generating answers.
2.  **The Intern (Fine-Tuning)**: A Llama-3-8B model fine-tuned (QLoRA) on a domain-specific Q/A dataset to internalize the knowledge and tone of a financial analyst.

This repository contains the complete pipeline: Data Factory (ingestion/synthesis), Fine-Tuning, RAG retrieval, and an "Evaluation Arena" to benchmark both approaches on Accuracy (ROUGE, LLM-as-a-Judge), Latency, and Cost.

---

##  Features

*   **Intelligent Data Factory**: 
    *   Parses PDFs into cleaner text/markdown.
    *   Chunks data semantically.
    *   Generates a synthetic "Golden Test Set" (Q/A pairs) using LLMs for training and evaluation.
*   **Advanced RAG Pipeline**:
    *   **Vector Database**: Weaviate (supports Cloud & Local).
    *   **Search**: Hybrid Search (Sparse BM25 + Dense Vectors).
    *   **Reranking**: Post-retrieval reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    * **Citations**: Returns source page numbers with answers.
*   **Fine-Tuning Workflow**:
    *   Efficient fine-tuning of Llama-3-8B using QLoRA (4-bit quantization).
    *   Custom instruction formatting for financial analysis tone.
*   **Evaluation Arena**:
    *   **ROUGE-L**: Measures textual overlap with ground truth.
    *   **LLM-as-a-Judge**: Uses GPT-4o to score answer quality (1-5) and provide reasoning.
    *   **ROI Analysis**: Calculates cost per 1k queries to determine the most business-viable solution.

---

##  Project Structure

```
financial-intelligence-engine/
├── data/
│   ├── raw/                  # Original PDFs (e.g., uber_2024_ar.pdf)
│   ├── processed/            # Chunked JSONs, Golden Test Sets
│   └── results/              # Evaluation metrics and logs
├── notebooks/
│   ├── 01_data_factory.ipynb       # Ingestion, Chunking, & Q/A Generation
│   ├── 02_finetuning_intern.ipynb  # QLoRA Fine-tuning of Llama-3
│   ├── 03_rag_librarian.ipynb      # RAG Pipeline Setup (Weaviate)
│   └── 04_evaluation_arena.ipynb   # Head-to-head comparison
├── src/
│   ├── config/
│   │   └── config.yaml       # Central configuration (Models, API Keys, Paths)
│   ├── services/
│   │   └── llm_services.py   # LLM Factory (OpenAI, Groq, OpenRouter)
│   └── utils/
│   │   └── cost_tracker.py   # Token usage and cost calculation
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

##  Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/financial-intelligence-engine.git
cd financial-intelligence-engine
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
1.  **Environment Variables**: Create a `.env` file in the root directory (see `.env.example`).
    ```ini
    OPENAI_API_KEY=sk-...
    GROQ_API_KEY=gsk_...
    OPENROUTER_API_KEY=sk-or-...
    HF_TOKEN=hf_...
    WCS_URL=https://... (Optional for Weaviate Cloud)
    WCS_API_KEY=...     (Optional for Weaviate Cloud)
    ```
2.  **App Config**: Modify `src/config/config.yaml` to select your preferred models (e.g., switch between `gpt-4o`, `llama-3`, etc.).

---

##  Usage Guide

The project is structured as a series of Jupyter Notebooks to be run in order.

### Step 1: The Data Factory (`01_data_factory.ipynb`)
Run this notebook to parse the PDF financial report. It will chunk the text and use an LLM to generate a synthetic dataset of Question-Answer pairs (`train.jsonl` and `golden_test_set.jsonl`).

### Step 2: The Intern (`02_finetuning_intern.ipynb`)
*Requires GPU (Colab T4 or better recommended).*
This notebook loads the `train.jsonl` dataset and fine-tunes a Llama-3-8B model to learn the specific knowledge and style of the report.

### Step 3: The Librarian (`03_rag_librarian.ipynb`)
Sets up the Weaviate vector database. It ingests the chunks created in Step 1, generates embeddings, and demonstrates the Hybrid Search + Reranking pipeline.

### Step 4: The Evaluation Arena (`04_evaluation_arena.ipynb`)
The final showdown! This notebook runs the `golden_test_set.jsonl` against both "The Intern" (or its proxy) and "The Librarian". It outputs a side-by-side comparison of:
*   **Accuracy**: Did it answer correctly?
*   **Latency**: How fast was it?
*   **Cost**: How much did it cost?

---

##  Evaluation Metrics

We use a multi-faceted evaluation approach:

| Metric | Description |
| :--- | :--- |
| **ROUGE-L** | Measures word overlap (good for exact phrasing). |
| **LLM Judge** | An impartial "Judge" model (GPT-4o) scores answers 1-5 based on reasoning. |
| **Latency** | End-to-end response time in seconds. |
| **Cost** | Dollar cost per query based on input/output tokens. |

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built by Sahas Induwara**
