import streamlit as st
import requests
from repo_engine import CodebaseRAGPipeline

st.set_page_config(
    page_title="GitHub Repo Codebase Search Engine", page_icon="💻", layout="wide"
)

if "code_pipeline" not in st.session_state:
    st.session_state.code_pipeline = CodebaseRAGPipeline()
if "code_history" not in st.session_state:
    st.session_state.code_history = []


def get_ollama_models():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=2)
        if res.status_code == 200:
            models = [m["model"] for m in res.json().get("models", [])]
            if models:
                return models
    except:
        pass
    return ["llama3.2:3b", "qwen3.5:2b"]


st.title("💻 GitHub Repository Codebase Search Engine")
st.caption("AST-Aware Semantic Code Navigation & Local Q&A Studio")

with st.sidebar:
    st.header("⚙️ Codebase Config")
    top_k = st.slider("Top Code Chunks", 1, 8, 4)

    models = get_ollama_models()
    selected_model = st.selectbox("Generator Model", models)

    st.divider()
    if st.button("🗑️ Reset Code Database", use_container_width=True):
        st.session_state.code_pipeline.clear_database()
        st.session_state.code_history = []
        st.success("Code index cleared!")

tab_chat, tab_upload, tab_inspector = st.tabs(
    ["💬 Code Q&A Assistant", "📁 Repository Uploader", "🔬 AST Inspector"]
)

with tab_upload:
    st.subheader("Ingest Repository Source Files")
    uploaded_files = st.file_uploader(
        "Upload Python/Code files",
        type=["py", "txt", "js", "ts"],
        accept_multiple_files=True,
    )

    if st.button("Parse & Index Codebase", type="primary"):
        if uploaded_files:
            file_map = {}
            for f in uploaded_files:
                file_map[f.name] = f.getvalue().decode("utf-8")

            with st.spinner("Parsing AST nodes and computing vector embeddings..."):
                stats = st.session_state.code_pipeline.index_repository_files(file_map)

            st.success(
                f"Successfully indexed {len(file_map)} files into {stats['total_chunks']} AST structural chunks in {stats['indexing_time_sec']}s!"
            )
        else:
            st.warning("Please upload at least one code file.")

with tab_chat:
    st.subheader("Chat with your Codebase")

    for msg in st.session_state.code_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📂 View Cited Code Snippets"):
                    for src in msg["sources"]:
                        st.code(src.page_content, language="python")
                        st.caption(
                            f"File: {src.metadata.get('source')} | Symbol: {src.metadata.get('name', 'N/A')}"
                        )

    if query := st.chat_input(
        "Ask a question about code implementation, logic, or functions..."
    ):
        st.session_state.code_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching AST index & querying local LLM..."):
                res = st.session_state.code_pipeline.answer_code_query(
                    query, top_k=top_k, model_name=selected_model
                )
                answer = res["answer"]
                sources = res["sources"]

                st.markdown(answer)
                if sources:
                    with st.expander("📂 View Cited Code Snippets"):
                        for src in sources:
                            st.code(src.page_content, language="python")
                            st.caption(
                                f"File: {src.metadata.get('source')} | Symbol: {src.metadata.get('name', 'N/A')}"
                            )
                st.caption(f"⏱️ Latency: {res['latency']}s")

        st.session_state.code_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

with tab_inspector:
    st.subheader("Extracted AST Syntax Nodes")
    if st.session_state.code_pipeline.indexed_nodes:
        for idx, node in enumerate(st.session_state.code_pipeline.indexed_nodes):
            with st.expander(
                f"🔹 {node.get('type', 'chunk').upper()}: `{node.get('name', 'block')}` ({node.get('source', '')})"
            ):
                st.write("**Docstring:**", node.get("docstring", "N/A"))
                st.write(
                    "**Line Range:**",
                    f"Lines {node.get('start_line', '?')} - {node.get('end_line', '?')}",
                )
    else:
        st.info("No repository files indexed yet.")
