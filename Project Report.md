# Engineering Report: Comparative Evaluation of Hybrid RAG Architecture for Financial Document Intelligence

**Date:** February 14, 2026  
**Author:** Sahas Induwara  
**Student Number:** 0200  
**Subject:** Technical Assessment of Retrieval-Augmented Generation (RAG) vs. Fine-Tuned LLM in Financial Contexts

---

## 1. Executive Summary

This report details a comparative evaluation of two distinct architectures for automated financial document analysis: a **Hybrid Retrieval-Augmented Generation (RAG)** system (referred to as "The Librarian") and a standalone **Large Language Model (LLM)** baseline (referred to as "The Intern").

The primary objective was to determine the optimal architecture for querying complex financial disclosures, specifically Annual Reports (10-K/10-Q filings), where precision and verifiable accuracy are paramount.

### Key Findings

The evaluation results indicate that the **Hybrid RAG architecture significantly outperforms** the baseline LLM in tasks requiring quantitative precision and strict adherence to source documents.

- **Accuracy:** The RAG system achieved a mean **LLM-Judge Score of 3.83/5.0**, surpassing the baseline's **3.50/5.0**.
- **Reliability:** The RAG system consistently provided verifiable citations and correctly identified when information was absent, whereas the baseline model frequently generated "hallucinated" figures.
- **Latency:** The precision of the RAG system comes with a latency trade-off (13.9s vs. 4.8s), which is deemed acceptable for back-office analytical workflows where correctness outweighs speed.

**Conclusion:** The Hybrid RAG architecture is the recommended solution for regulated financial environments due to its superior auditability and hallucination resistance.

---

## 2. Methodology

The evaluation employed a rigorous "Evaluation Arena" framework to benchmark both systems against a "Golden Set" of ground-truth Q&A pairs.

### 2.1 Experimental Setup

#### Dataset Generation

A curated dataset was extracted from Uber Technologies Inc. Annual Reports (2019-2023).

- **Total Samples:** 100 Q&A pairs.
- **Diverse Query Types:** Including quantitative extraction (e.g., net income figures), qualitative risk assessment, and "negative constraint" questions designed to test the model's ability to refuse unanswerable queries.

#### System Architectures

| Feature                | Configuration A: Hybrid RAG ("Librarian")                                    | Configuration B: Baseline LLM ("Intern")                     |
| :--------------------- | :--------------------------------------------------------------------------- | :----------------------------------------------------------- |
| **Model**              | `gemini-1.5-flash` (Temperature: 0.2)                                        | `gpt-4o-mini` (Temperature: 0.1)                             |
| **Knowledge Source**   | **External Vector Database** (Weaviate) containing specific document chunks. | **Internal Parametric Memory** only (pre-trained knowledge). |
| **Retrieval Strategy** | Hybrid Search (Sparse BM25 + Dense Vector).                                  | N/A (Direct generation).                                     |
| **Reranking**          | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) used to rank top 5 chunks.          | N/A.                                                         |
| **Prompt Engineering** | Role-based with strict grounding constraints ("Answer only from context").   | Standard instruction-following.                              |

---

## 3. Quantitative Analysis

The performance of both systems was measured across four key metrics.

### 3.1 Performance Metrics Table

| Metric                     | Hybrid RAG | Baseline LLM | Delta     | Interpretation                                                                                  |
| :------------------------- | :--------- | :----------- | :-------- | :---------------------------------------------------------------------------------------------- |
| **Avg. Judge Score (1-5)** | **3.83**   | 3.50         | +0.33     | RAG answers are consistently rated higher for accuracy and completeness by the LLM Judge.       |
| **Avg. ROUGE-L Score**     | **0.35**   | 0.16         | +0.19     | RAG answers share significantly more vocabulary (exact numbers/phrasing) with the Ground Truth. |
| **Avg. Latency**           | 13.92s     | **4.79s**    | +9.13s    | The retrieval and reranking steps introduce a necessary processing overhead.                    |
| **Avg. Cost per Query**    | $0.00019   | **$0.00014** | +$0.00005 | RAG incurs slightly higher token costs due to injecting retrieved context into the prompt.      |

**Analysis:**
The significant advantage in **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)** for the RAG system confirms its superiority in retrieving precise terminology and figures. The Baseline model, while fluent, tends to paraphrase or generalize, resulting in lower lexical overlap with the verified ground truth. The latency increase is linear with the number of retrieved chunks but remains within acceptable bounds for analytical tools.

---

## 4. The "Hallucination" Audit: Qualitative Case Studies

A manual audit of the responses reveals critical failure modes in the Baseline LLM that render it unsuitable for unsupervised financial analysis.

### Case Study 4.1: Quantitative Hallucination

**Context:** The user asks for a specific financial figure from the 2024 projections.

> **Query:** _"How much of the foreign net operating loss carryforwards is set to expire in 2024?"_

| System           | Response                                                                             | Assessment                                                                                     |
| :--------------- | :----------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **Ground Truth** | _"$759 million"_                                                                     |                                                                                                |
| **Hybrid RAG**   | **"$759 million"**                                                                   | **PASS:** The model retrieved the exact paragraph containing this figure.                      |
| **Baseline LLM** | _"Loss carryforwards typically expire after 20 years, depending on jurisdiction..."_ | **FAIL:** The model provided a generic textbook definition instead of the specific data point. |

**Impact:** In financial modeling, a generic answer is useless, while a hallucinated number would introduce material error. RAG resolves this by grounding the generation in retrieved evidence.

---

### Case Study 4.2: Semantic Hallucination (The "Plausible Lie")

**Context:** The user asks about specific regulated products mentioned in the text.

> **Query:** _"What additional products, besides Mobility, are mentioned as being subject to legal regulations?"_

| System           | Response                                                                | Assessment                                                                                                           |
| :--------------- | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Ground Truth** | _"Delivery and Freight products."_                                      |                                                                                                                      |
| **Hybrid RAG**   | **"Delivery and Freight products."**                                    | **PASS:** Correctly identified the specific business lines mentioned in the report.                                  |
| **Baseline LLM** | _"Healthcare products, financial services, and consumer technology..."_ | **FAIL:** The model hallucinated plausible but incorrect industries based on general knowledge of regulated sectors. |

**Impact:** This failure mode is particularly dangerous because the answer sounds authoritative and logically consistent, making it difficult for human reviewers to detect without cross-referencing valid documents.

---

### Case Study 4.3: Compliance Safety (Negative Constraints)

**Context:** The user asks a question about a topic _not_ present in the report.

> **Query:** _"What types of legislative proposals are currently being considered regarding driver status?"_

| System           | Response                                                         | Assessment                                                                                                                 |
| :--------------- | :--------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| **Ground Truth** | _Information not found in context._                              |                                                                                                                            |
| **Hybrid RAG**   | **"I don't have enough information."**                           | **PASS:** The model correctly identified that the document did not contain this specific information.                      |
| **Baseline LLM** | _Lists detailed legislative proposals (PRO Act, ABC Test, etc.)_ | **FAIL:** The model retrieved external knowledge that may be outdated or irrelevant to the specific report being analyzed. |

**Impact:** For compliance audits, identifying missing information is as critical as finding existing information. The Baseline model's tendency to answer every question creates a liability risk.

---

## 5. Conclusion & Recommendations

### 5.1 Architectural Recommendation is Hybrid RAG

Based on the empirical evidence presented, the **Hybrid RAG architecture is recommended** for deployment.

- It eliminates the "Black Box" nature of standard LLMs by providing **traceable citations**.
- It allows for **instantaneous updates** (ingesting a new PDF effectively "retrains" the system in seconds).
- It significantly reduces **operational risk** by defaulting to silence rather than fabrication when data is missing.

### 5.2 Future Optimizations

To further enhance reliability and usability, the following improvements are proposed:

1.  **Metadata Pre-filtering:** Restricting search scope by year (e.g., `WHERE year = '2023'`) to prevent data contamination across reporting periods.
2.  **Specialized Tabular Parsing:** Integrating visual document understanding models (e.g., LlamaParse) to better interpret complex financial tables, which are often poorly represented in plain text.
3.  **Agentic Routing:** Implementing a routing layer to direct general queries to a cheaper model and complex financial queries to the RAG pipeline, optimizing cost-efficiency.

---

_End of Report_
