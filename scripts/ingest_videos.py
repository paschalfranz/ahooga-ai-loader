import os
import csv
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict
 
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
 
load_dotenv()
 
# -------------------------
# ENV
# -------------------------
 
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")
 
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
 
# -------------------------
# PATHS
# -------------------------
 
CSV_PATH = Path("video_metadata.csv")
SUMMARIES_ROOT = Path("video_summaries")
 
 
def validate_env():
    required = {
        "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
        "AZURE_SEARCH_ADMIN_KEY": SEARCH_KEY,
        "AZURE_SEARCH_INDEX": INDEX_NAME,
        "AZURE_OPENAI_ENDPOINT": OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": OPENAI_KEY,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
 
 
validate_env()
 
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY),
)
 
openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=OPENAI_ENDPOINT,
)
 
# -------------------------
# HELPERS
# -------------------------
 
def normalize_header(value: str) -> str:
    return str(value).strip().lower()
 
 
def normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:
    normalized = {}
    for k, v in row.items():
        if k is None:
            continue
        normalized[normalize_header(k)] = v
    return normalized
 
 
def load_video_metadata(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
 
    metadata = {}
 
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
 
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
 
        reader = csv.DictReader(f, delimiter=delimiter)
 
        if reader.fieldnames is None:
            raise ValueError("CSV file has no headers.")
 
        reader.fieldnames = [normalize_header(h) for h in reader.fieldnames]
 
        if "file_name" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'file_name' column. Found headers: {reader.fieldnames}")
 
        for row in reader:
            row = normalize_row_keys(row)
 
            file_name = str(row.get("file_name", "")).strip()
            if not file_name:
                print(f"SKIP: Row with empty file_name: {row}")
                continue
 
            metadata[file_name] = row
 
    return metadata
 
 
def clean_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
 
 
def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
 
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_chars:
                current = para
            else:
                start = 0
                while start < len(para):
                    end = start + max_chars
                    chunks.append(para[start:end])
                    start = end - overlap if end - overlap > start else end
                current = ""
 
    if current:
        chunks.append(current)
 
    return chunks
 
 
def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding
 
 
def deterministic_id(source_file: str, chunk_id: int, chunk: str) -> str:
    payload = f"{source_file}|{chunk_id}|{chunk}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
 
 
def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
 
 
def summary_file_to_video_file(summary_path: Path) -> str:
    return f"{summary_path.stem}.mp4"
 
 
def build_document_record(meta: Dict[str, str], source_file: str, chunk: str, chunk_id: int) -> Dict:
    return {
        "id": deterministic_id(source_file, chunk_id, chunk),
        "content": chunk,
        "source_file": source_file,
        "chunk_id": chunk_id,
        "product": meta.get("product", "unknown"),
        "system_supplier": meta.get("system_supplier", "Unknown"),
        "system_generation": meta.get("system_generation", "unknown"),
        "component": meta.get("component", "general"),
        "doc_type": meta.get("doc_type", "unknown"),
        "compatibility": meta.get("compatibility", "unknown"),
        "manual_version": "na",
        "audience": meta.get("audience", "both"),
        "drivetrain_type": meta.get("drivetrain_type", "unknown"),
        "source_type": meta.get("source_type", "video"),
        "has_speech": parse_bool(meta.get("has_speech", "false")),
        "has_subtitles": parse_bool(meta.get("has_subtitles", "false")),
        "is_continuation": parse_bool(meta.get("is_continuation", "false")),
        "related_video": meta.get("related_video", ""),
        "content_vector": embed_text(chunk),
    }
 
 
def process_summary_file(summary_path: Path, metadata_lookup: Dict[str, Dict[str, str]]) -> List[Dict]:
    source_file = summary_file_to_video_file(summary_path)
 
    if source_file not in metadata_lookup:
        print(f"SKIP: No CSV metadata for summary {summary_path.name}")
        return []
 
    meta = metadata_lookup[source_file]
 
    text = summary_path.read_text(encoding="utf-8")
    text = clean_text(text)
 
    if not text:
        print(f"WARNING: Empty summary file: {summary_path.name}")
        return []
 
    chunks = chunk_text(text)
    docs = []
 
    for i, chunk in enumerate(chunks, start=1):
        docs.append(build_document_record(meta, source_file, chunk, i))
 
    print(f"{summary_path.name}: {len(chunks)} chunks")
    return docs
 
 
def upload_documents(documents: List[Dict], batch_size: int = 50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = search_client.upload_documents(documents=batch)
        success = sum(1 for r in result if r.succeeded)
        print(f"Uploaded {success}/{len(batch)} documents")
 
 
def main():
    if not SUMMARIES_ROOT.exists():
        print(f"Summary folder not found: {SUMMARIES_ROOT.resolve()}")
        return
 
    metadata_lookup = load_video_metadata(CSV_PATH)
    summary_files = sorted(SUMMARIES_ROOT.glob("*.txt"))
 
    if not summary_files:
        print(f"No summary files found in: {SUMMARIES_ROOT.resolve()}")
        return
 
    all_docs: List[Dict] = []
 
    for summary_path in summary_files:
        all_docs.extend(process_summary_file(summary_path, metadata_lookup))
 
    print(f"Total video chunks to upload: {len(all_docs)}")
 
    if all_docs:
        upload_documents(all_docs)
 
 
if __name__ == "__main__":
    main()