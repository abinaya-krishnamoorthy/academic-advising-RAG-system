import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Run: streamlit run app.py
from pathlib import Path
import concurrent.futures as cf
import subprocess
import re
import streamlit as st

from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ---------- helpers ----------
def format_docs(docs):
    """Format retrieved documents for context display."""
    if not docs:
        return ""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        sheet = meta.get("sheet")
        row = meta.get("row_index")
        tag = f"{src}" + (f" — sheet={sheet}, row={row}" if sheet is not None else "")
        parts.append(f"[Source: {tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def with_timeout(fn, seconds=120):
    """Run function with timeout; handle slow LLMs gracefully."""
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except cf.TimeoutError:
            raise TimeoutError("Model exceeded timeout limit.")


def detect_ollama_models():
    """Detect available Ollama models locally."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        models = []
        for line in result.stdout.splitlines():
            if line.strip() and not line.startswith("NAME"):
                name = line.split()[0]
                models.append(name)
        return models
    except Exception:
        return []


# ---------- deterministic instructor-course extractor ----------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
COURSE_RE = re.compile(r"\b([A-Z]{2,4}\s?\d{3,4}[A-Z]?)\b")
TERM_ALIASES = {"fall 2025": ["fall 2025", "2025 fall", "fa 2025", "f25", "fall-2025", "autumn 2025"]}


def text_has_term(text: str, term: str) -> bool:
    t = term.lower().strip()
    aliases = TERM_ALIASES.get(t, [t])
    lt = text.lower()
    return any(a in lt for a in aliases)


def extract_people_answer(docs, person_name: str, term: str):
    """Extract instructor courses + email directly from retrieved text."""
    name_variants = {person_name}
    parts = person_name.split()
    if len(parts) == 2:
        first, last = parts
        name_variants.add(f"{first[0]}. {last}")
        name_variants.add(f"{first} {last[0]}.")
        name_variants.add(last)

    courses = set()
    email = None
    matched_sources = []

    for d in docs:
        meta = d.metadata or {}
        txt = f"{d.page_content}\n"
        ltxt = txt.lower()

        if not any(v.lower() in ltxt for v in name_variants):
            continue
        if term and not text_has_term(ltxt, term):
            pass  # soft term match

        if email is None:
            m = EMAIL_RE.search(txt)
            if m:
                email = m.group(0)

        for m in COURSE_RE.finditer(txt):
            code = m.group(1).upper().replace("  ", " ")
            line = txt[m.start(): txt.find("\n", m.start()) if "\n" in txt[m.start():] else len(txt)]
            title = None
            dash_idx = line.find(" - ")
            if dash_idx != -1:
                title = line[dash_idx + 3:].strip()
            mtitle = re.search(r"(course[_\s]?title|title)\s*:\s*([^|]+)", line, flags=re.I)
            if mtitle:
                title = mtitle.group(2).strip()
            courses.add(f"{code}" + (f" — {title}" if title else ""))

        matched_sources.append(meta)

    return {"email": email, "courses": sorted(courses), "sources": matched_sources}


# ---------- UI ----------
st.set_page_config(page_title="Academic Advising Assistant", layout="wide")
st.title("🎓 Academic Advising Assistant")

st.markdown("""
This assistant answers questions about **degree plans, course requirements, instructors, and academic policies**
using your `.docx`, `.pdf`, `.xlsx`, and `.csv` files indexed in Chroma.
""")

with st.expander("Example questions"):
    st.markdown("""
- What are the required courses for the Business Analytics minor?
- How many credits are needed to complete the Management major?
- What courses does Ziyi Wang teach in Fall 2025, and what is his email?
- Are there prerequisites for MKTG 4050?
""")

# ---------- sidebar ----------
st.sidebar.header("Settings")
persist_dir = st.sidebar.text_input(
    "Chroma persist directory",
    value=r"C:\temp\chroma_full_nomic"
)
p = Path(persist_dir)
st.sidebar.write("Resolved path:", p.resolve())
st.sidebar.write("Exists:", p.exists())
st.sidebar.write("Has chroma.sqlite3:", (p / "chroma.sqlite3").exists())

k = st.sidebar.slider("Top-k retrieved chunks", 2, 12, 6, 1)
search_type = st.sidebar.selectbox("Search type", ["mmr", "similarity"], index=0)
fetch_k = st.sidebar.slider("MMR fetch_k (if MMR)", min_value=max(10, k), max_value=100, value=25)
lambda_mult = st.sidebar.slider("MMR diversity λ (if MMR)", 0.0, 1.0, 0.5, 0.05)
retrieval_only = st.sidebar.checkbox("Answer with retrieval only (no LLM)", value=False)

# Auto-detect Ollama models
available_models = detect_ollama_models()
preferred_order = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b-instruct", "llama3.2"]
default_model = next((m for m in preferred_order if m in available_models), preferred_order[0])

model_name = st.sidebar.selectbox(
    "Ollama LLM (auto-detected)",
    options=available_models or preferred_order,
    index=(available_models.index(default_model) if default_model in available_models else 0),
    help="Automatically selects the fastest available model (like llama3.2:1b)."
)
st.sidebar.caption(f"Detected Ollama models: {', '.join(available_models) or 'none found'}")

embed_model = st.sidebar.selectbox(
    "Embedding model (must match index)",
    ["nomic-embed-text", "mxbai-embed-large", "bge-m3"],
    index=0
)

# ---------- chain builder ----------
@st.cache_resource(show_spinner=False)
def create_chain(persist_directory: str, k: int, search_type: str,
                 model_name: str, embed_model: str,
                 fetch_k: int, lambda_mult: float):
    """Build retrieval + LLM chain (cached for performance)."""
    p = Path(persist_directory)
    if not p.exists() or not (p / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"Chroma directory not found or empty: {persist_directory}")

    embedding_fn = OllamaEmbeddings(model=embed_model)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_fn)

    if search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    try:
        col_count = vectorstore._collection.count()
    except Exception:
        try:
            data = vectorstore.get()
            col_count = len(data.get("ids", [])) if isinstance(data, dict) else None
        except Exception:
            col_count = None


    template = """You are a knowledgeable academic advising assistant.
Use only the provided context to answer accurately and concisely.
If an answer is not in the context, say "That information is not available."

# Context
{context}

# Question
{question}

# Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=2048,      # larger context window
        num_predict=256    # allows longer, complete answers
    )

    get_context = RunnableLambda(lambda q: format_docs(retriever.invoke(q)))

    chain = (
        {"context": get_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, col_count
try:
    chain, retriever, col_count = create_chain(
        persist_dir, k, search_type, model_name, embed_model, fetch_k, lambda_mult
    )
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------- diagnostics ----------
with st.sidebar.expander("Diagnostics", expanded=True):
    st.write(f"Chroma path: `{persist_dir}`")
    st.write(f"Embedding model: `{embed_model}`")
    st.write(f"LLM: `{model_name}`")
    st.write(f"Top-k: {k}, Search: {search_type}")
    st.write(f"Indexed chunks: **{col_count if col_count else 'unknown'}**")
    test_q = st.text_input("Test retrieval-only query", value="List required courses for the Business Analytics minor")
    if st.button("Run retrieval test"):
        docs = retriever.invoke(test_q)
        st.write(f"Retrieved {len(docs)} chunk(s).")
        st.code(format_docs(docs)[:2000] or "(no context returned)")

# ---------- chat ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can help you explore degree plans and instructor details."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about courses, degree plans, or instructors:")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching..."):
        docs = retriever.invoke(question)

    # --- deterministic extractor for instructor lookups ---
    lower_q = (question or "").lower()
    if ("teach" in lower_q or "course" in lower_q) and ("fall 2025" in lower_q or "2025 fall" in lower_q or "f25" in lower_q):
        person_name = "Ziyi Wang"  # you can make this dynamic if needed
        term = "fall 2025"
        extracted = extract_people_answer(docs, person_name, term)

        if extracted["email"] or extracted["courses"]:
            answer_lines = [f"**Instructor:** {person_name}", f"**Term:** Fall 2025"]
            if extracted["courses"]:
                answer_lines.append("**Courses:**")
                for c in extracted["courses"]:
                    answer_lines.append(f"- {c}")
            else:
                answer_lines.append("**Courses:** Not found in the context.")
            answer_lines.append(f"**Email:** {extracted['email'] or 'Not found in the context.'}")
            answer = "\n".join(answer_lines)

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.expander("Sources"):
                for i, d in enumerate(docs, 1):
                    meta = d.metadata or {}
                    title = meta.get("source") or "(untitled)"
                    st.markdown(f"**{i}. {title}**")
                    st.caption(d.page_content[:400] + ("..." if len(d.page_content) > 400 else ""))
            st.stop()
    # --- end deterministic extractor ---

    if retrieval_only:
        answer = "Top matches:\n\n" + "\n\n".join(
            f"- **{(d.metadata or {}).get('source', '?')}**: {d.page_content[:300]}…" for d in docs
        )
    else:
        with st.spinner("Thinking..."):
            try:
                answer = with_timeout(lambda: chain.invoke(question), seconds=90)
            except TimeoutError:
                st.warning("Model took too long. Showing retrieved context instead.")
                answer = "Top matches:\n\n" + "\n\n".join(
                    f"- **{(d.metadata or {}).get('source', '?')}**: {d.page_content[:300]}…" for d in docs
                )

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
