from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import traceback
import json
import uuid
from pathlib import Path

from ask import answer_question
from routes.tickets import router as tickets_router

app = FastAPI(title="Ahooga Internal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets only if they exist
assets_dir = Path("static/assets")
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


class AskResponse(BaseModel):
    answer: str
    conversation_id: str


def safe_extract(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return value.get("text", "")
    return str(value)


def normalize_request_payload(raw):
    if not isinstance(raw, dict):
        return {}

    data = raw

    if "body" in raw:
        body = raw["body"]

        if isinstance(body, str):
            try:
                data = json.loads(body)
            except Exception:
                data = {}
        elif isinstance(body, dict):
            data = body
        else:
            data = {}

    return data if isinstance(data, dict) else {}


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path("static/index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Ahooga Internal API is running</h1>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request):
    try:
        body_bytes = await request.body()

        if not body_bytes:
            conversation_id = str(uuid.uuid4())
            return AskResponse(
                answer="I didn’t receive your question. Please send it again.",
                conversation_id=conversation_id,
            )

        try:
            raw_text = body_bytes.decode("utf-8", errors="ignore").strip()
            raw = json.loads(raw_text) if raw_text else {}
        except Exception:
            raw = {}

        data = normalize_request_payload(raw)

        question = safe_extract(data.get("question")).strip()
        conversation_id = safe_extract(data.get("conversation_id")).strip()

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if not question:
            return AskResponse(
                answer="I didn’t receive your question. Please send it again.",
                conversation_id=conversation_id,
            )

        answer = answer_question(
            user_input=question,
            conversation_id=conversation_id
        )

        return AskResponse(
            answer=answer,
            conversation_id=conversation_id,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {str(e)}"
        )


app.include_router(tickets_router)