# USPSTF Medical RAG

**Evidence-Grounded Clinical Question Answering with Hybrid Retrieval and Safety-Aware Generation**

A Retrieval-Augmented Generation (RAG) system for answering clinical questions using evidence retrieved from **U.S. Preventive Services Task Force (USPSTF) guidelines**.

The system combines semantic and lexical retrieval with grounded LLM generation to reduce unsupported answers and provide traceable clinical evidence.

> **Research prototype:** This system is intended for experimentation and clinical AI research. It is not a substitute for professional medical advice or clinical judgment.

---

## Overview

Medical question answering requires more than generating fluent responses. The underlying information needs to be:

* Relevant to the user's question
* Grounded in trusted clinical sources
* Traceable to supporting evidence
* Protected against unsupported or out-of-domain generation

This project implements a complete RAG pipeline that addresses these requirements through **hybrid retrieval, rank fusion, evidence-grounded generation, and safety checks**.

---

## Architecture

```text
                         User Query
                             │
                             ▼
                    ┌─────────────────┐
                    │   Input Guard   │
                    │ Validation &    │
                    │ Safety Checks   │
                    └────────┬────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Hybrid Retrieval    │
                 │                       │
                 │ ┌─────────┐ ┌───────┐ │
                 │ │ Dense   │ │ BM25  │ │
                 │ │ Search  │ │ Search│ │
                 │ └────┬────┘ └───┬───┘ │
                 │      └─────┬─────┘     │
                 │            ▼           │
                 │      RRF Fusion         │
                 └────────────┬────────────┘
                              │
                              ▼
                       Top-K Evidence
                              │
                              ▼
                    ┌─────────────────┐
                    │ Gemini Generator│
                    │ Grounded Prompt │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Output Guard   │
                    │ Safety &        │
                    │ Grounding Check │
                    └────────┬────────┘
                             │
                             ▼
                    Answer + Sources
```

---

## Key Features

### Hybrid Retrieval

The system combines two complementary retrieval strategies:

* **Dense retrieval** using `BAAI/bge-small-en-v1.5` embeddings and ChromaDB
* **Sparse retrieval** using BM25Okapi for exact medical terminology, keywords, and acronyms
* **Reciprocal Rank Fusion (RRF)** to combine the two retrieval rankings

This helps balance semantic similarity with exact lexical matching.

### Grounded Generation

Google Gemini generates responses using only the retrieved clinical evidence.

The generation layer is configured with strict grounding instructions to reduce unsupported claims and provide source-aware answers.

### Safety Layer

The pipeline includes:

* Input validation
* Prompt-injection protection
* Retrieval score gating
* Out-of-domain detection
* Safe refusal behavior
* Medical disclaimer handling
* Output validation

When sufficient evidence cannot be retrieved from the USPSTF dataset, the system can refuse to generate an unsupported clinical answer.

---

## Technology Stack

| Component        | Technology                                       |
| ---------------- | ------------------------------------------------ |
| LLM              | Google Gemini                                    |
| Embeddings       | `BAAI/bge-small-en-v1.5`                         |
| Vector Database  | ChromaDB                                         |
| Sparse Retrieval | BM25Okapi                                        |
| Retrieval Fusion | Reciprocal Rank Fusion (RRF)                     |
| Language         | Python                                           |
| Evaluation       | Custom retrieval & generation evaluation scripts |

---

## Project Structure

```text
Medical_RAG/
│
├── data/
│   ├── raw/
│   │   └── uspstf/                  # Source clinical guideline PDFs
│   │
│   └── processed/
│       ├── chroma_db/               # Persistent vector store
│       └── chunks.json              # Processed chunks + metadata
│
├── scripts/
│   ├── evaluate_retrieval.py        # Retrieval benchmark
│   └── evaluate_generation.py       # Generation & safety evaluation
│
├── src/
│   ├── generation/
│   │   └── generator.py             # Gemini generation layer
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # Dense + BM25 + RRF
│   │   └── vector_store.py          # ChromaDB & embeddings
│   │
│   └── safety/
│       ├── input_guard.py           # Input validation & injection checks
│       ├── medical_disclaimer.py    # Medical disclaimer handling
│       └── output_guard.py          # Response safety validation
│
├── tests/
├── docs/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Example

### Input

```text
What is the recommended daily dosage of folic acid to prevent neural tube defects?
```

### Pipeline

```text
Question
   ↓
Input validation
   ↓
Dense retrieval + BM25
   ↓
RRF ranking
   ↓
Top-K USPSTF evidence
   ↓
Gemini grounded generation
   ↓
Safety validation
   ↓
Answer + cited sources
```

### Output

The generated response is accompanied by source metadata such as:

* Document
* Clinical topic
* Recommendation grade
* Page
* Publication year

This allows the generated answer to be traced back to the retrieved evidence.

---

## Evaluation

The current retrieval benchmark evaluates whether relevant clinical evidence is retrieved within the top-K results.

### Current Results

| Metric                                                | Result |
| ----------------------------------------------------- | -----: |
| Precision@3                                           | 93.33% |
| Precision@5 — baseline                                | 94.67% |
| Precision@5 — after filtering zero-score BM25 results | 96.00% |
| Out-of-domain refusal on tested queries               |   100% |

> Evaluation results are based on the current test set and should not be interpreted as clinical performance metrics.

---

## Engineering Decisions

### Why Hybrid Retrieval?

Medical queries often contain both:

* Semantic concepts that benefit from embedding-based retrieval
* Exact terminology, abbreviations, and clinical keywords that benefit from lexical search

Combining dense retrieval with BM25 allows the system to capture both types of relevance.

### Failure Analysis: Sparse Retrieval Noise

During evaluation, BM25 could return zero-score chunks when no exact query terms were found.

These irrelevant results could enter the Top-K retrieval window and introduce noise.

The retrieval pipeline was updated to filter BM25 results with:

```python
score > 0.0
```

This improved the measured Precision@5 from **94.67% to 96.00%** on the current evaluation set.

### Out-of-Domain Safety

A major failure mode in RAG systems is allowing the LLM to answer questions that are not supported by the underlying knowledge base.

The system therefore uses retrieval-score gating and strict generation constraints.

When sufficient evidence is unavailable, the system returns:

```text
The provided USPSTF guidelines do not contain sufficient evidence to answer this question.
```

This prevents the generator from relying solely on its pretrained knowledge for unsupported queries.

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/naira404/Medical_RAG.git
cd Medical_RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=your_configured_gemini_model
```

---

## Using the RAG Pipeline

```python
from src.retrieval.hybrid_retriever import MedicalHybridRetriever
from src.generation.generator import MedicalGenerator
from src.safety.medical_disclaimer import append_disclaimer

retriever = MedicalHybridRetriever(
    vector_store_dir="data/processed/chroma_db",
    chunks_json_path="data/processed/chunks.json",
    collection_name="uspstf_guidelines"
)

generator = MedicalGenerator()

query = "What is the recommended daily dosage of folic acid to prevent neural tube defects?"

contexts = retriever.retrieve(
    query=query,
    top_k=3,
    alpha=0.6
)

result = generator.generate_response(
    query=query,
    contexts=contexts
)

final_response = append_disclaimer(result["answer"])
```

---

## Limitations

This project has several important limitations:

* The knowledge base is limited to the indexed USPSTF guidelines.
* Retrieval and generation performance depends on the quality and coverage of the evaluation dataset.
* Evaluation metrics are based on a limited test set.
* The system is not validated for real-world clinical deployment.
* Generated responses must be reviewed by qualified healthcare professionals.

---

## Disclaimer

This project is an experimental research prototype for exploring Retrieval-Augmented Generation in healthcare.

It is **not a medical device, diagnostic system, or substitute for professional medical advice**.

Clinical recommendations generated by the system must be independently verified against authoritative sources and reviewed by a qualified healthcare professional.
