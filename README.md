# Neural Network & Machine Learning Playground

Welcome to the **Neural Network & Machine Learning Playground**! This repository is a curated collection of Jupyter notebooks implementing foundational Deep Learning algorithms from scratch, alongside practical NLP applications—ranging from Retrieval-Augmented Generation (RAG) pipelines to Semantic Search engines and custom web scrapers.

---

## 🗺️ Project Navigation

Below is a summary of the notebooks and resources available in this folder:

| Component / Notebook | Description | Core Stack |
| :--- | :--- | :--- |
| [llm-from-scratch.ipynb](llm-from-scratch.ipynb) | Custom GPT-like Large Language Model built entirely from scratch. | PyTorch, Regex, Python |
| [mnist-from-scratch.ipynb](mnist-from-scratch.ipynb) | 2-Layer Neural Network trained on MNIST digits using raw matrix operations. | NumPy, Pandas, Matplotlib |
| [rag_pipeline.ipynb](rag_pipeline.ipynb) | Hybrid-search RAG pipeline on Chess Laws with Cross-Encoder reranking. | LangChain, ChromaDB, BM25, SentenceTransformers |
| [embeddings_creator.ipynb](embeddings_creator.ipynb) | Keyphrase extraction and vector database population from movie plot files. | KeyBERT, SentenceTransformers, GPU-accelerated |
| [semantic_movie_recommender.ipynb](semantic_movie_recommender.ipynb) | Similarity-based search engine recommending movies from natural language queries. | SentenceTransformers, NumPy, Cosine Similarity |
| [webscraping.ipynb](webscraping.ipynb) | Custom review parser designed to scrape user feedback from Letterboxd. | Cloudscraper, BeautifulSoup |

---

## 🛠️ Detailed Breakdown

### 🧠 1. Scratch Implementations

#### 📖 GPT from Scratch (`llm-from-scratch.ipynb`)
An educational walkthrough implementing a complete generative decoder-only Transformer model:
* **Custom Tokenizer**: Implements a simple BPE/Regex tokenizer to encode and decode raw text.
* **Architecture**: Builds the Embedding layer, Positional Encodings, Multi-Head Causal Attention mechanism, feed-forward blocks, and layer normalizations.
* **Training**: Trains on `verdict.txt` to learn token transition patterns and generate text.

#### 🔢 Neural Network from Scratch (`mnist-from-scratch.ipynb`)
Trains a network to classify handwritten digits without any automated gradient computation library:
* **Linear Algebra**: Uses manual matrix multiplications and NumPy arrays.
* **Math Derivatives**: Calculates forward pass outputs and handles manual backpropagation equations, including standard ReLU and Softmax derivatives.
* **Performance**: Reaches >84% accuracy in just 500 gradient descent iterations.

---

### 🔍 2. Natural Language Processing & Search Applications

#### ♟️ Chess Laws RAG Pipeline (`rag_pipeline.ipynb`)
An advanced Retrieval-Augmented Generation (RAG) system parsing the FIDE Laws of Chess:
* **Document Extraction**: Employs `pymupdf4llm` to preserve structured headers from `LawsOfChess.pdf`.
* **Logical Splitter**: Splits sections cleanly using markdown headings, appending breadcrumbs to contextualize deep rules sections.
* **Hybrid Search**: Fuses dense semantic retrievals (via ChromaDB) and sparse term matching (via BM25Okapi).
* **Reranking**: Filters candidate contexts using a deep Cross-Encoder model (`sentence-transformers/all-MiniLM-L6-v2`) to match relevant rules blocks for questions.

#### 🎬 Keyphrase Extraction & Embeddings (`embeddings_creator.ipynb` & `semantic_movie_recommender.ipynb`)
A semantic recommendation engine working on a database of movies:
1. **Extraction**: Cleans movie plots (`dataset/movie1.csv`) and extracts top keyword candidates with KeyBERT on CUDA.
2. **Dense Vector Mapping**: Encodes text to 384-dimensional space and exports numerical representations as `movie_embeddings.npy`.
3. **Recommender**: Evaluates user semantic queries (e.g., *"A horror movie involving possession and exorcism"*) against precomputed embeddings using cosine similarity to return matches.

#### 🕸️ Cloud-Bypassing Web Scraper (`webscraping.ipynb`)
A light utility script to bypass modern bot checks:
* **Bypass**: Utilizes `cloudscraper` to configure headers resembling real Chrome browsers.
* **Parsing**: Collects and structures text bodies containing Letterboxd user reviews.

---

## 📊 Supporting Files & Assets

* **[verdict.txt](verdict.txt)**: Source text corpus (Edith Wharton's short story *The Verdict*) used to train the custom LLM tokenizer.
* **[LawsOfChess.pdf](LawsOfChess.pdf)**: Official rulebook from FIDE used as the input source for the Chess RAG pipeline.
* **[chess.json](chess.json)**: Extracted JSON data structure preserving section metadata.
* **`chroma_asl_db/`**: Local SQLite-backed Chroma database directory storing vector index mappings.
* **`dataset/movie1.csv`**: A dense CSV dataset containing movie metadata (titles, years, and plot summaries).

---

## ⚙️ Requirements & Local Setup

### Prerequisite Dependencies
To run these notebooks locally, you will need to install the following Python packages:

```bash
pip install torch numpy pandas matplotlib scikit-learn \
            pymupdf4llm langchain langchain-text-splitters \
            langchain-community langchain-huggingface chromadb \
            langchain-chroma rank_bm25 sentence-transformers \
            keybert tqdm cloudscraper beautifulsoup4
```

*Note: For GPU acceleration in `embeddings_creator.ipynb` and `llm-from-scratch.ipynb`, a CUDA-capable GPU is highly recommended.*
