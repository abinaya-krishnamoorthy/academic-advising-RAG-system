# build_index_single.py
from pathlib import Path
from datetime import datetime, time
import re
import numpy as np
import pandas as pd
from math import ceil

from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------- CONFIG ----------
BASE_DIR    = Path(r"C:\Users\abich\OneDrive - The University of Colorado Denver\My streamlit app")
RAW_DIR     = BASE_DIR / "data" / "raw"                
PERSIST_DIR = Path(r"C:\temp\chroma_full_nomic")  # new folder
EMBED_MODEL = "nomic-embed-text"

print("RAW_DIR:", RAW_DIR.resolve())
print("PERSIST_DIR:", PERSIST_DIR.resolve())

# ---------- Excel helpers for cleaning data and converts rows to text and wraps it in Langchain Document ----------
def normalize_header(s: str) -> str:
    s = "" if s is None else str(s)
    s = " ".join(s.strip().split())
    s = s.replace("/", " ").replace("-", " ").replace(".", " ")
    s = re.sub(r"\s+", " ", s)
    return "_".join([t for t in s.lower().split(" ") if t])

def clean_cell(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and v.strip() == ""):
        return ""
    if pd.isna(v): return ""
    if isinstance(v, (pd.Timestamp, datetime)):
        if v.time() == time(0, 0): return v.strftime("%Y-%m-%d")
        return v.strftime("%Y-%m-%d %I:%M %p").lstrip("0")
    if isinstance(v, (pd.Timedelta, pd.Period)): return str(v)
    if isinstance(v, (int, np.integer)): return str(v)
    if isinstance(v, (float, np.floating)): return str(int(v)) if float(v).is_integer() else f"{v:g}"
    return str(v).strip()

def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        val_str = clean_cell(val)
        if not val_str: continue
        label = col.replace("_", " ").title()
        parts.append(f"{label}: {val_str}")
    return " | ".join(parts)

def load_excel_as_row_docs(excel_path: Path) -> list[Document]:
    docs: list[Document] = []
    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet, dtype=object)
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty: continue

        if df.columns.tolist() == list(range(len(df.columns))):
            candidate_idx = None
            for i in range(min(5, len(df))):
                if df.iloc[i].notna().mean() >= 0.5:
                    candidate_idx = df.index[i]; break
            if candidate_idx is not None:
                df.columns = [normalize_header(c) for c in df.loc[candidate_idx].tolist()]
                df = df.drop(index=candidate_idx)
            else:
                df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        else:
            df.columns = [normalize_header(c) for c in df.columns]

        for c in df.columns:
            df[c] = df[c].map(clean_cell)

        df = df.replace("", np.nan).dropna(axis=0, how="all").fillna("")
        if df.empty: continue

        df = df.rename(columns={k: v for k, v in {"title":"course_title","instructor_name":"instructor"}.items() if k in df.columns})

        for ridx, row in df.iterrows():
            text = row_to_text(row)
            if not text: continue
            docs.append(Document(
                page_content=text,
                metadata={"source": excel_path.name, "sheet": sheet, "row_index": int(ridx) if isinstance(ridx,(int,np.integer)) else str(ridx), "doc_type": "excel"},
            ))
    return docs

# ---------- Loaders & splitters for Excel, docs and Pdf ----------
def load_pdf(fp: Path):
    docs = PyPDFLoader(str(fp)).load()
    for d in docs: d.metadata.update({"source": fp.name, "doc_type": "pdf"})
    return docs

def split_pdf(docs):
    return RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=50).split_documents(docs)

def load_docx(fp: Path):
    docs = Docx2txtLoader(str(fp)).load()
    for d in docs: d.metadata.update({"source": fp.name, "doc_type": "docx"})
    return docs

def split_docx(docs):
    out = []
    for d in docs:
        text = d.page_content
        headers = [("#","h1"),("##","h2"),("###","h3")]
        md_docs = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=False).split_text(text)
        inner = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
        pieces = inner.split_documents(md_docs or [Document(page_content=text, metadata=d.metadata)])
        for c in pieces: c.metadata.update(d.metadata)
        out.extend(pieces)
    return out

def split_excel(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, separators=[" | ","\n"," ",""]).split_documents(docs)

def load_csv_as_row_docs(csv_path: Path) -> list[Document]:
    docs = []
    df = pd.read_csv(csv_path, dtype=object).fillna("")
    df.columns = [normalize_header(c) for c in df.columns]
    for _, row in df.iterrows():
        text = row_to_text(row)
        if text:
            docs.append(Document(page_content=text, metadata={"source": csv_path.name, "sheet":"CSV","doc_type":"csv"}))
    return docs

def split_csv(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, separators=[" | ","\n"," ",""]).split_documents(docs)

# ---------- batched vector database builder. ----------
def build_batched(chunks, persist_path: Path, embed_model: str, batch_size=200):
    persist_path.mkdir(parents=True, exist_ok=True)
    emb = OllamaEmbeddings(model=embed_model)
    vs = Chroma(persist_directory=str(persist_path), embedding_function=emb)

    total = len(chunks)
    num_batches = ceil(total / batch_size)
    print(f"Indexing {total} chunks in {num_batches} batches of {batch_size}...")

    for i in range(0, total, batch_size):
        j = min(i + batch_size, total)
        vs.add_documents(chunks[i:j])
        try:
            count = len(vs.get().get('ids', []))
        except Exception:
            count = None
        print(f"  ✓ Batch {i//batch_size + 1}/{num_batches} ({j}/{total}) — store count: {count}")

    print("Done.")
    return vs

# ---------- Building Main index code----------
def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    files = [p for p in RAW_DIR.glob("*.*") if not p.name.startswith("~$")]
    print("RAW files found:", [p.name for p in files])
    if not files:
        print("RAW_DIR is empty — put your PDFs/DOCX/XLSX/CSV in", RAW_DIR)
        return

    all_chunks = []
    pdf_chunks = docx_chunks = xlsx_chunks = csv_chunks = 0

    for fp in files:
        ext = fp.suffix.lower()
        try:
            if ext == ".pdf":
                pcs = split_pdf(load_pdf(fp)); all_chunks += pcs; pdf_chunks += len(pcs)
                print(f"[pdf]   {fp.name}: +{len(pcs)}")
            elif ext == ".docx":
                dcs = split_docx(load_docx(fp)); all_chunks += dcs; docx_chunks += len(dcs)
                print(f"[docx]  {fp.name}: +{len(dcs)}")
            elif ext in (".xlsx", ".xls"):
                ecs = split_excel(load_excel_as_row_docs(fp)); all_chunks += ecs; xlsx_chunks += len(ecs)
                print(f"[excel] {fp.name}: +{len(ecs)}")
            elif ext == ".csv":
                ccs = split_csv(load_csv_as_row_docs(fp)); all_chunks += ccs; csv_chunks += len(ccs)
                print(f"[csv]   {fp.name}: +{len(ccs)}")
            else:
                print(f"[skip]  {fp.name} (unsupported)")
        except Exception as e:
            print(f"[skip]  {fp.name}: {e}")

    print("---- Chunk totals ----")
    print("PDF  :", pdf_chunks)
    print("DOCX :", docx_chunks)
    print("EXCEL:", xlsx_chunks)
    print("CSV  :", csv_chunks)
    print("ALL  :", len(all_chunks))

    if not all_chunks:
        print("No chunks created. Check file types/extensions and loaders.")
        return

    vs = build_batched(all_chunks, PERSIST_DIR, EMBED_MODEL, batch_size=200)
    try:
        count = len(vs.get().get("ids", []))
    except Exception:
        count = None
    print("Index built at:", PERSIST_DIR.resolve())
    print("Total ids in store:", count if count is not None else "(unknown)")

if __name__ == "__main__":
    main()
