# 📚 Academic Papers RAG

A Retrieval-Augmented Generation (RAG) system that lets you **ask questions about a collection of academic papers** and get answers grounded in the actual text, not in the model's memory.

Built with **PyMuPDF**, **Sentence-Transformers**, **FAISS**, **Groq**, and **Streamlit**.

---

## ✨ Features

- **PDF ingestion**: extracts text from research papers and splits it into chunks.
- **Semantic search**: embeds chunks with Sentence-Transformers and stores them in a FAISS index for fast similarity search.
- **Grounded answers**: the most relevant chunks are sent to an LLM (via Groq) as context, so answers come from the papers.
- **Interactive UI**: a Streamlit chat app to ask questions.
- **Built-in evaluation**: a script that runs a set of test questions and saves detailed predictions, error analysis, and summary results.

---

## 🏗️ How It Works

```
 papers/*.pdf
     │
     ▼
 ingest.py ──► chunks.json          (extract text + split into chunks)
     │
     ▼
 build_index.py ──► faiss_index.bin (embed chunks + build vector index)
     │
     ▼
 rag_pipeline.py                    (query → retrieve top-k chunks → LLM answer)
     │
     ├──► app.py                    (Streamlit chat interface)
     └──► evaluate.py               (run eval_questions.json, save metrics)
```

1. **Ingest**: PDFs in `papers/` are read with PyMuPDF and split into text chunks, saved to `chunks.json`.
2. **Index**: each chunk is converted into an embedding and added to a FAISS index (`faiss_index.bin`).
3. **Retrieve**: at question time, the query is embedded and the closest chunks are retrieved.
4. **Generate**: the retrieved chunks + the question are sent to an LLM on Groq, which writes the final answer.

---

## 📁 Project Structure

| Path | Description |
|---|---|
| `papers/` | Source PDF papers |
| `ingest.py` | Extracts text from PDFs and creates chunks |
| `build_index.py` | Builds embeddings and the FAISS index |
| `rag_pipeline.py` | Core RAG logic: retrieval + answer generation |
| `app.py` | Streamlit web app |
| `evaluate.py` | Evaluation script |
| `eval_questions.json` | Evaluation questions |
| `chunks.json` | Generated text chunks |
| `faiss_index.bin` | Generated FAISS vector index |
| `detailed_predictions.csv` | Per-question predictions from evaluation |
| `error_analysis.csv` | Analysis of failed or weak answers |
| `test_results_summary.csv` | Summary of test results |
| `validation_comparison.csv` | Comparison of validation runs / configurations |
| `requirements.txt` | Python dependencies |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/domaamohamed00-ui/academic-papers-rag.git
cd academic-papers-rag

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY="your_api_key_here"      # On Windows (PowerShell): $env:GROQ_API_KEY="your_api_key_here"
```

> ⚠️ Never commit your API key to GitHub.

---

## ▶️ Usage

### 1. Add your papers
Put your PDF files inside the `papers/` folder.

### 2. Process the papers and build the index

```bash
python ingest.py
python build_index.py
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`) and start asking questions.

### 4. (Optional) Run the evaluation

```bash
python evaluate.py
```

Results are saved to CSV files in the project root (`detailed_predictions.csv`, `error_analysis.csv`, `test_results_summary.csv`).

---

## 📊 Evaluation

The project includes an evaluation workflow to measure answer quality:

- **`eval_questions.json`**: a set of questions about the papers.
- **`detailed_predictions.csv`**: the system's answer for each question.
- **`error_analysis.csv`**: where and why the system failed.
- **`test_results_summary.csv`** and **`validation_comparison.csv`**: overall results and comparisons between runs.

### Validation: chunk size comparison (`validation_comparison.csv`)

| Configuration | Recall@4 | Answer Correctness | Citation Correctness | Abstention Accuracy | Unsupported Claim Rate |
|---|---|---|---|---|---|
| `chunk_size=250` | 1.0 | 0.75 | 0.143 | 1.0 | 0.0 |
| `chunk_size=500` | 1.0 | **1.00** | 0.000 | 1.0 | 0.0 |

`chunk_size=500` gave the best answer correctness on validation, so it was selected for the final test.

### Final held-out test (`test_results_summary.csv`)

| Configuration | Recall@4 | Answer Correctness | Abstention Accuracy | Unsupported Claim Rate |
|---|---|---|---|---|
| Held-out test (`chunk_size=500`) | 1.0 | 0.75 | 1.0 | 0.0 |

### Key takeaways

- **Retrieval:** Recall@4 is 1.0 in all runs, so the right chunk was always found in the top 4 results.
- **Safety:** Abstention accuracy is 1.0 and the unsupported claim rate is 0.0, so the system declined to answer when the papers had no answer and did not invent claims.
- **Answer quality:** Answer correctness is 0.75 on the held-out test.
- **Known limitation:** Citation correctness is low (0.143 and 0.000 on validation), so source citations still need improvement. See `error_analysis.csv` for details.

> Note: the evaluation set is small, so these numbers are indicative, not a final benchmark.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Embeddings | [Sentence-Transformers](https://www.sbert.net/) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| LLM inference | [Groq](https://groq.com/) |
| Data handling | pandas, NumPy |
| UI | [Streamlit](https://streamlit.io/) |

---

## 🔮 Future Improvements

- Show source citations (paper name + page) next to each answer
- Add support for uploading new papers directly from the UI
- Experiment with hybrid search (keyword + semantic) and re-ranking
- Try different chunk sizes and embedding models

---

## 👤 Author

**Adham Mohamed**
B.Sc. Computer Science & AI, Pharos University, Alexandria, Egypt

GitHub: [@domaamohamed00-ui](https://github.com/domaamohamed00-ui)

---

## 📄 License

Add a license (for example MIT) if you want others to use or build on this project.
