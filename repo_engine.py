import os
import ast
import time
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate


class CodebaseRAGPipeline:
    def __init__(
        self,
        collection_name: str = "codebase_rag",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_code_db",
    ):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.indexed_nodes: List[Dict[str, Any]] = []

    def parse_python_ast(self, file_path: str, file_content: str) -> List[Document]:
        """Parses Python code into AST-aware chunks (Functions and Classes)."""
        documents = []
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = getattr(node, "end_lineno", start_line + len(node.body))
                    node_code = "\n".join(lines[start_line - 1 : end_line])
                    docstring = ast.get_docstring(node) or "No docstring provided."

                    metadata = {
                        "source": file_path,
                        "type": (
                            "class" if isinstance(node, ast.ClassDef) else "function"
                        ),
                        "name": node.name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "docstring": docstring,
                    }

                    documents.append(
                        Document(page_content=node_code, metadata=metadata)
                    )

            # If no functions/classes found, fallback to standard splitting
            if not documents:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                documents = splitter.create_documents(
                    [file_content], metadatas=[{"source": file_path, "type": "file"}]
                )

        except Exception:
            # Fallback for syntax errors or non-parseable code
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = splitter.create_documents(
                [file_content], metadatas=[{"source": file_path, "type": "raw"}]
            )

        return documents

    def index_repository_files(self, file_dict: Dict[str, str]) -> Dict[str, Any]:
        """Indexes dictionary of {file_path: file_content} into vector store."""
        start_time = time.time()
        all_docs = []

        for path, content in file_dict.items():
            if path.endswith(".py"):
                docs = self.parse_python_ast(path, content)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=50
                )
                docs = splitter.create_documents(
                    [content], metadatas=[{"source": path, "type": "file"}]
                )

            all_docs.extend(docs)

        if all_docs:
            self.vector_store.add_documents(all_docs)
            for d in all_docs:
                self.indexed_nodes.append(d.metadata)

        elapsed = round(time.time() - start_time, 3)
        return {
            "total_chunks": len(all_docs),
            "indexing_time_sec": elapsed,
        }

    def answer_code_query(
        self, query: str, top_k: int = 4, model_name: str = "llama3.2:3b"
    ) -> Dict[str, Any]:
        """Performs semantic search across codebase and prompts Ollama."""
        start_time = time.time()

        retrieved_docs = self.vector_store.similarity_search(query, k=top_k)

        context_str = "\n\n---\n\n".join(
            [
                f"File: `{doc.metadata.get('source')}` | Symbol: `{doc.metadata.get('name', 'file')}` (Lines {doc.metadata.get('start_line', 'N/A')}-{doc.metadata.get('end_line', 'N/A')}):\n```python\n{doc.page_content}\n```"
                for doc in retrieved_docs
            ]
        )

        system_prompt = """You are an expert Codebase Navigator and Senior Software Engineer. Answer the developer's question accurately using ONLY the provided code snippets. Cite the exact file names and functions/classes referenced.

Code Context:
{context}

Question: {query}

Answer:"""

        prompt = ChatPromptTemplate.from_template(system_prompt)
        formatted_prompt = prompt.format(context=context_str, query=query)

        try:
            llm = Ollama(model=model_name, temperature=0.1)
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            response = f"Local LLM Error: {str(e)}"

        latency = round(time.time() - start_time, 3)
        return {
            "answer": response,
            "sources": retrieved_docs,
            "formatted_prompt": formatted_prompt,
            "latency": latency,
        }

    def clear_database(self):
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="codebase_rag",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        self.indexed_nodes = []
