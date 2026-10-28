This is a **GitHub Repository Codebase Search Engine & Local Q&A Studio** with **AST (Abstract Syntax Tree) Awareness**.

![Summary](https://github.com/user-attachments/assets/6c92a7ec-5009-4f6d-bfc2-18b4886f27df)

Unlike standard RAG systems that chop text into arbitrary fixed-size chunks (which often split a function or class in half), this application intelligently parses source code at the logical level—extracting individual functions and classes, their docstrings, and exact line ranges. Coupled with local embedding models and **Ollama**, it lets you query your codebase, debug logic, and explore repository architecture completely offline with 100% data privacy.

---

## Project Structure

```text
ast-rag-engine/
├── repo_engine.py              
├── code_app.py       
├── requirements.txt    
└── README.md           

```

---

## How the Two Files Work Together

The project follows a clean separation of concerns:

1. **`repo_engine.py` (The Backend Code Intelligence Engine):** Handles AST code parsing, embedding generation, Chroma vector storage (`./chroma_code_db`), and Ollama LLM code reasoning.
2. **`code_app.py` (The Frontend UI Studio):** Built with **Streamlit**, it provides the web interface for uploading repository files, chatting with your code, configuring search settings, and inspecting extracted AST structural nodes.

---

## Deep Dive: File-by-File Breakdown

### 1. `repo_engine.py` (Backend Engine)

* **AST Code Parser (`parse_python_ast`):** Uses Python's native `ast` module to walk through `.py` files. It identifies class (`ast.ClassDef`) and function (`ast.FunctionDef`) nodes, extracts their exact line number boundaries (`start_line` to `end_line`), captures docstrings, and packages them into rich metadata. If a file is non-Python or has syntax errors, it gracefully falls back to character splitting.
* **Repository Ingestion (`index_repository_files`):** Takes a dictionary of file paths and contents, routes Python files through the AST parser and other files through standard text splitters, and adds all structural chunks into ChromaDB while logging metadata.
* **Code Q&A Orchestration (`answer_code_query`):** Performs a semantic similarity search across the codebase vector store, formats the retrieved code snippets into markdown code blocks with file paths and line numbers, and prompts the local Ollama LLM using a senior software engineer persona to ensure accurate, cited answers.
* **Database Reset (`clear_database`):** Wipes the codebase vector collection and resets internal state trackers.

### 2. `code_app.py` (Frontend UI)

* **Session State & Model Discovery:** Persists the pipeline instance and chat history across UI reruns, and dynamically pings local Ollama endpoints to populate the model selection dropdown.
* **Repository Uploader Tab:** Accepts multiple code files (`.py`, `.js`, `.ts`, `.txt`), sends them to the backend engine, and displays indexing telemetry (file counts, structural chunks created, and indexing speed).
* **Code Q&A Assistant Tab:** Provides a chat interface where developers can ask questions about implementation logic or architecture, rendering responses alongside expandable source code snippets and latency metrics.
* **AST Inspector Tab:** Acts as a code structure browser, letting you inspect every extracted function or class node, view its docstring, and check its exact line range in the repository.

---

### 🛠️ Execution Instructions

1. **Install Prerequisites**:
Ensure Python 3.10+ is installed on your local machine.
2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


3. **Install & Start Ollama (Local LLM)**:
* Download Ollama from [ollama.com](https://ollama.com).
* Pull your preferred model:
```bash
ollama pull llama3

```




4. **Run the Streamlit Application**:
```bash
streamlit run app.py

```
