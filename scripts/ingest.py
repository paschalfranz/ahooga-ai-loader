import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import pytesseract
from PIL import Image

load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

DOCS_ROOT = Path(os.getenv("DOCS_ROOT", "."))
IMAGES_ROOT = Path(os.getenv("IMAGES_ROOT", "images"))


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

FILE_OVERRIDES: Dict[str, Dict[str, str]] = {
    "max_manual_old-owner_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "general",
        "doc_type": "manual",
        "compatibility": "bafang-only",
        "manual_version": "old",
        "audience": "customer",
        "drivetrain_type": "nexus",
    },
    "max_manual_new-owner_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Mivice",
        "system_generation": "new",
        "component": "general",
        "doc_type": "manual",
        "compatibility": "mivice-only",
        "manual_version": "new",
        "audience": "customer",
        "drivetrain_type": "alfine-derailleur",
    },
    "max_assembly_bafang-controller_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "controller",
        "doc_type": "assembly",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_assembly_bafang-torque-sensor_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "torque-sensor",
        "doc_type": "assembly",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_assembly_bafang-torque-sensor_v2.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "torque-sensor",
        "doc_type": "assembly",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_assembly_both-ffi_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Both",
        "system_generation": "both",
        "component": "ffi",
        "doc_type": "assembly",
        "compatibility": "both",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "both",
    },
    "max_assembly_both-front-clasp_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Both",
        "system_generation": "both",
        "component": "front-clasp",
        "doc_type": "assembly",
        "compatibility": "both",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "both",
    },
    "max_assembly_full-bike_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Unknown",
        "system_generation": "unknown",
        "component": "general",
        "doc_type": "assembly",
        "compatibility": "unknown",
        "manual_version": "na",
        "audience": "customer",
        "drivetrain_type": "unknown",
    },
    "max_troubleshooting_mivice-powertrain_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Mivice",
        "system_generation": "new",
        "component": "powertrain",
        "doc_type": "troubleshooting",
        "compatibility": "mivice-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "alfine-derailleur",
    },
    "max_troubleshooting_bafang-powertrain_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "powertrain",
        "doc_type": "troubleshooting",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_part_numbers_bafang_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "general",
        "doc_type": "parts",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_system_app_bafang_goplus_v3_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "app",
        "doc_type": "system",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_system_display_bafang_dpe180-e181_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "display",
        "doc_type": "system",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_system_display_mivice_d101_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Mivice",
        "system_generation": "new",
        "component": "display",
        "doc_type": "system",
        "compatibility": "mivice-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "alfine-derailleur",
    },
    "max_system_tool_bafang_besst_v2024_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "tool",
        "doc_type": "system",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_system_update_bafang_goplus_steps_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Bafang",
        "system_generation": "old",
        "component": "update",
        "doc_type": "system",
        "compatibility": "bafang-only",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "nexus",
    },
    "max_warranty_dealer_v1.pdf": {
        "product": "MAX",
        "system_supplier": "Both",
        "system_generation": "both",
        "component": "warranty",
        "doc_type": "warranty",
        "compatibility": "both",
        "manual_version": "na",
        "audience": "dealer",
        "drivetrain_type": "both",
    },
}


def normalize_filename(filename: str) -> str:
    name = filename.strip()
    while name.lower().endswith(".pdf.pdf"):
        name = name[:-4]
    return name


def extract_text_from_image(image_path: Path) -> str:
    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {e}")
        return ""


def extract_text_from_pdf(pdf_path: Path) -> str:
    parts: List[str] = []
    pdf_image_dir = IMAGES_ROOT / pdf_path.stem
    pdf_image_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                parts.append(text)

            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]

                    image_filename = f"page_{page_num}_img_{img_index}.png"
                    image_path = pdf_image_dir / image_filename

                    with open(image_path, "wb") as img_file:
                        img_file.write(image_data)

                    text_from_image = extract_text_from_image(image_path)
                    if text_from_image:
                        parts.append(text_from_image)

                except Exception as e:
                    print(
                        f"Error extracting OCR image from {pdf_path.name}, "
                        f"page {page_num}, image {img_index}: {e}"
                    )

    return "\n".join(parts).strip()


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


def parse_from_filename(filename: str) -> Dict[str, str]:
    base = normalize_filename(filename).lower()
    if base.endswith(".pdf"):
        base = base[:-4]

    normalized = base.replace("-", "_")

    fields = {
        "product": "MAX" if normalized.startswith("max_") else "unknown",
        "system_supplier": "Unknown",
        "system_generation": "unknown",
        "component": "general",
        "doc_type": "unknown",
        "compatibility": "unknown",
        "manual_version": "na",
        "audience": "both",
        "drivetrain_type": "unknown",
    }

    if "_manual_" in normalized:
        fields["doc_type"] = "manual"
    elif "_assembly_" in normalized:
        fields["doc_type"] = "assembly"
    elif "_troubleshooting_" in normalized:
        fields["doc_type"] = "troubleshooting"
    elif "_part_" in normalized or "_parts_" in normalized:
        fields["doc_type"] = "parts"
    elif "_system_" in normalized:
        fields["doc_type"] = "system"
    elif "_warranty_" in normalized:
        fields["doc_type"] = "warranty"

    if "bafang" in normalized:
        fields["system_supplier"] = "Bafang"
        fields["system_generation"] = "old"
        fields["compatibility"] = "bafang-only"
    elif "mivice" in normalized:
        fields["system_supplier"] = "Mivice"
        fields["system_generation"] = "new"
        fields["compatibility"] = "mivice-only"
    elif "both" in normalized:
        fields["system_supplier"] = "Both"
        fields["system_generation"] = "both"
        fields["compatibility"] = "both"

    for comp in [
        "controller",
        "torque_sensor",
        "front_clasp",
        "ffi",
        "powertrain",
        "display",
        "app",
        "tool",
        "update",
        "warranty",
        "general",
    ]:
        if comp in normalized:
            fields["component"] = comp.replace("_", "-")
            break

    if "old_owner" in normalized:
        fields["manual_version"] = "old"
    elif "new_owner" in normalized:
        fields["manual_version"] = "new"

    return fields


def merge_metadata(filename: str) -> Dict[str, str]:
    lookup = normalize_filename(filename)
    meta = parse_from_filename(lookup)
    meta.update(FILE_OVERRIDES.get(lookup, {}))
    return meta


def embed_text(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding


def deterministic_id(filename: str, chunk_id: int, chunk: str) -> str:
    payload = f"{normalize_filename(filename)}|{chunk_id}|{chunk}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_document_record(filename: str, chunk: str, chunk_id: int) -> Dict:
    normalized_filename = normalize_filename(filename)
    meta = merge_metadata(normalized_filename)

    return {
        "id": deterministic_id(normalized_filename, chunk_id, chunk),
        "content": chunk,
        "source_file": normalized_filename,
        "chunk_id": chunk_id,
        "product": meta["product"],
        "system_supplier": meta["system_supplier"],
        "system_generation": meta["system_generation"],
        "component": meta["component"],
        "doc_type": meta["doc_type"],
        "compatibility": meta["compatibility"],
        "manual_version": meta["manual_version"],
        "audience": meta["audience"],
        "drivetrain_type": meta["drivetrain_type"],
        "content_vector": embed_text(chunk),
    }


def process_pdf(pdf_path: Path) -> List[Dict]:
    filename = normalize_filename(pdf_path.name)
    raw_text = extract_text_from_pdf(pdf_path)
    text = clean_text(raw_text)

    if not text:
        print(f"WARNING: No extractable text in {filename}")
        return []

    chunks = chunk_text(text)
    docs = []

    for i, chunk in enumerate(chunks, start=1):
        docs.append(build_document_record(filename, chunk, i))

    print(f"{filename}: {len(chunks)} chunks")
    return docs


def upload_documents(documents: List[Dict], batch_size: int = 50):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = search_client.upload_documents(documents=batch)
        success = sum(1 for r in result if r.succeeded)
        print(f"Uploaded {success}/{len(batch)} documents")


def main():
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DOCS_ROOT.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {DOCS_ROOT}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    all_docs: List[Dict] = []

    for pdf_path in pdf_files:
        all_docs.extend(process_pdf(pdf_path))

    print(f"Total chunks to upload: {len(all_docs)}")
    if all_docs:
        upload_documents(all_docs)


if __name__ == "__main__":
    main()