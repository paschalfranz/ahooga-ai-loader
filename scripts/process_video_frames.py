import os
import csv
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# -------------------------
# CONFIG
# -------------------------

FRAMES_ROOT = Path("video_frames")
CSV_PATH = Path("video_metadata.csv")
OUTPUT_ROOT = Path("video_summaries")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Increased from 12 to improve subtitle/tool coverage
MAX_FRAMES_PER_VIDEO = 25

# Use this to test only a few videos first
ONLY_PROCESS = None

# -------------------------
# CLIENT
# -------------------------

def validate_env():
    required = {
        "AZURE_OPENAI_ENDPOINT": OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": OPENAI_KEY,
        "AZURE_OPENAI_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

validate_env()

client = AzureOpenAI(
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


def list_frame_files(video_frame_dir: Path) -> List[Path]:
    return sorted(video_frame_dir.glob("*.jpg"))


def choose_representative_frames(frame_files: List[Path], max_frames: int = 25) -> List[Path]:
    if len(frame_files) <= max_frames:
        return frame_files

    # Always include first and last
    selected = [frame_files[0], frame_files[-1]]

    slots = max_frames - 2
    if slots <= 0:
        return selected

    step = (len(frame_files) - 1) / (slots + 1)

    for i in range(1, slots + 1):
        idx = round(i * step)
        idx = max(0, min(idx, len(frame_files) - 1))
        selected.append(frame_files[idx])

    # Deduplicate while preserving order
    unique = []
    seen = set()
    for p in selected:
        if p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


def image_to_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def bool_str(value: str) -> str:
    return str(value).strip().lower()


def build_prompt(meta: Dict[str, str], frame_names: List[str]) -> str:
    return f"""
You are analyzing sampled frames from an Ahooga technical support video.

CRITICAL PRIORITY:
1. Extract ALL visible on-screen text, especially subtitles, tool lists, warnings, and short-lived text screens.
2. Then extract the visible mechanical actions and steps shown in the frames.

IMPORTANT RULES:
- Do NOT invent details that are not visible or strongly implied
- If something is unclear, say it is unclear
- Keep terminology consistent with the metadata
- This is an internal support knowledge extraction task
- The output should be useful for retrieval in a troubleshooting/assembly assistant
- If text appears briefly, still include it
- If text is partially visible, include a best-effort reconstruction and mention uncertainty
- Do NOT skip opening tool lists or early title screens

VIDEO METADATA:
file_name: {meta.get('file_name')}
product: {meta.get('product')}
system_supplier: {meta.get('system_supplier')}
system_generation: {meta.get('system_generation')}
component: {meta.get('component')}
doc_type: {meta.get('doc_type')}
source_type: {meta.get('source_type')}
compatibility: {meta.get('compatibility')}
audience: {meta.get('audience')}
drivetrain_type: {meta.get('drivetrain_type')}
has_speech: {meta.get('has_speech')}
has_subtitles: {meta.get('has_subtitles')}
is_continuation: {meta.get('is_continuation')}
related_video: {meta.get('related_video')}

SAMPLED FRAME FILES:
{chr(10).join("- " + x for x in frame_names)}

TASK:
Extract a structured summary with these sections exactly:

Title:
System:
Component:
Document type:
Source type:

Visible tools:
- ...

Visible on-screen text:
- ...

Observed steps:
1. ...
2. ...
3. ...

Warnings / important notes:
- ...

Confidence notes:
- Mention anything that is uncertain, partially visible, or inferred from limited frames.

At the end, add this final section exactly:
Retrieval summary:
A short retrieval-optimized paragraph that summarizes the procedure in plain text for semantic search.
"""


def summarize_video_from_frames(meta: Dict[str, str], frame_paths: List[Path]) -> str:
    prompt = build_prompt(meta, [p.name for p in frame_paths])

    content = [{"type": "text", "text": prompt}]
    for frame_path in frame_paths:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(frame_path)
            }
        })

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You extract structured technical knowledge from video frames. Be accurate, conservative, and concise."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def save_outputs(video_stem: str, meta: Dict[str, str], selected_frames: List[Path], summary_text: str):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    txt_path = OUTPUT_ROOT / f"{video_stem}.txt"
    json_path = OUTPUT_ROOT / f"{video_stem}.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    payload = {
        "video_stem": video_stem,
        "file_name": meta.get("file_name"),
        "metadata": meta,
        "selected_frames": [str(p) for p in selected_frames],
        "summary_text": summary_text,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


# -------------------------
# MAIN
# -------------------------

def main():
    metadata = load_video_metadata(CSV_PATH)

    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"Frames root not found: {FRAMES_ROOT}")

    video_dirs = [p for p in FRAMES_ROOT.iterdir() if p.is_dir()]
    if not video_dirs:
        print("No frame folders found.")
        return

    for video_dir in sorted(video_dirs):
        video_stem = video_dir.name
        file_name = f"{video_stem}.mp4"

        if ONLY_PROCESS and file_name not in ONLY_PROCESS:
            continue

        if file_name not in metadata:
            print(f"SKIP: No CSV metadata for {file_name}")
            continue

        frame_files = list_frame_files(video_dir)
        if not frame_files:
            print(f"SKIP: No frames found for {file_name}")
            continue

        selected_frames = choose_representative_frames(
            frame_files,
            max_frames=MAX_FRAMES_PER_VIDEO
        )

        print(f"\nProcessing: {file_name}")
        print(f"Using {len(selected_frames)} frames out of {len(frame_files)} total")

        try:
            summary_text = summarize_video_from_frames(metadata[file_name], selected_frames)
            save_outputs(video_stem, metadata[file_name], selected_frames, summary_text)
        except Exception as e:
            print(f"ERROR processing {file_name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()