import os
import asyncio
import logging
import uuid
from typing import List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from bs4 import BeautifulSoup

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройка БД ---
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Модели БД ---
class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    category = Column(String)
    is_live = Column(Boolean, default=True)
    tags = Column(JSON, default=[])
    stats = Column(JSON, default={"likes": 0, "inferences": 0})
    raw_data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

class SyncLog(Base):
    __tablename__ = "sync_logs"
    id = Column(Integer, primary_key=True)
    last_sync = Column(DateTime, default=datetime.utcnow)
    models_added = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="OpenGradient Catalog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем папку static
static_path = os.path.join(os.getcwd(), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    path = os.path.join(os.getcwd(), 'static', 'index.html')
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "index.html missing in /static"})

# Модели запросов
class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "General"

class ChatRequest(BaseModel):
    model_id: str
    message: str

# --- Логика Наполнения ---
async def scrape_opengradient_hub():
    url = "https://hub.opengradient.ai/models"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Referer": "https://hub.opengradient.ai/",
    }
    db: Session = SessionLocal()
    added_count = 0
    try:
        # Пытаемся парсить (если не заблокируют)
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    parts = [p for p in href.split('/') if p]
                    if len(parts) >= 3 and parts[0] == 'models':
                        m_id = f"{parts[1]}-{parts[2]}"
                        if not db.query(ModelDB).filter(ModelDB.id == m_id).first():
                            db.add(ModelDB(id=m_id, name=parts[2].title(), category="Hub", description="Imported from OpenGradient Hub", stats={"likes": 40, "inferences": 200}))
                            added_count += 1
                db.commit()

        # Гарантированные 16 моделей (включая stfu911)
        seeds = [
            {"id": "stfu911-task-deviation-corrector", "name": "Task Deviation Corrector", "desc": "Real-time correction for AI agent output variance."},
            {"id": "llama-3-8b-og", "name": "Llama 3 8B (OG)", "desc": "Meta Llama optimized for OpenGradient TEE."},
            {"id": "mistral-7b-v03", "name": "Mistral 7B v0.3", "desc": "Standard Mistral with extended 32k context."},
            {"id": "phi-3-mini", "name": "Phi-3 Mini", "desc": "Microsoft lightweight model for fast Edge inference."},
            {"id": "deepseek-coder", "name": "DeepSeek Coder 33B", "desc": "Advanced coding model for smart contract audits."},
            {"id": "gemma-7b", "name": "Gemma 7B IT", "desc": "Google DeepMind instruction-tuned model."},
            {"id": "neural-chat-7b", "name": "Neural Chat v3.3", "desc": "Intel-optimized model for logical reasoning."},
            {"id": "openhermes-2.5", "name": "OpenHermes 2.5", "desc": "Diverse fine-tune of Mistral for general tasks."},
            {"id": "starling-7b", "name": "Starling LM 7B", "desc": "RLHF-tuned model for superior chat performance."},
            {"id": "qwen-1.5-14b", "name": "Qwen 1.5 14B", "desc": "Alibaba's robust multilingual LLM."},
            {"id": "stable-code-3b", "name": "Stable Code 3B", "desc": "Efficient coding assistant for local dev."},
            {"id": "tinyllama-1.1b", "name": "TinyLlama 1.1B", "desc": "Ultra-compact model for mobile devices."},
            {"id": "solar-10.7b", "name": "Solar 10.7B", "desc": "Compact model with high reasoning capabilities."},
            {"id": "yi-34b-chat", "name": "Yi 34B Chat", "desc": "Powerful large-scale model from 01.AI."},
            {"id": "command-r", "name": "Command R", "desc": "Cohere's model specialized for RAG workflows."},
            {"id": "dolphin-mixtral", "name": "Dolphin Mixtral", "desc": "Uncensored high-performance MoE model."}
        ]
        for s in seeds:
            if not db.query(ModelDB).filter(ModelDB.id == s["id"]).first():
                db.add(ModelDB(id=s["id"], name=s["name"], description=s["desc"], category="Verified", is_live=True, stats={"likes": 150, "inferences": 4200}))
                added_count += 1
        db.commit()
        db.add(SyncLog(models_added=added_count))
        db.commit()
    except Exception as e:
        logger.error(f"Sync error: {e}")
    finally:
        db.close()

# --- API Endpoints ---
@app.get("/api/models")
def get_models():
    db = SessionLocal()
    return db.query(ModelDB).all()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not m: raise HTTPException(status_code=404)
    return m

@app.post("/api/chat")
async def chat(req: ChatRequest):
    responses = {
        "stfu911-task-deviation-corrector": "Detected task deviation: 0.05. Applying corrections via OpenGradient protocol...",
        "llama-3-8b-og": "Llama 3 is ready for inference in the TEE environment. Secure session established.",
    }
    reply = responses.get(req.model_id, f"Hello! This is a secure response from {req.model_id} on OG Network.")
    return {"reply": reply}

@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    total = db.query(ModelDB).count()
    last_log = db.query(SyncLog).order_by(SyncLog.id.desc()).first()
    return {"total_models": total, "live_models": total, "total_likes": total * 115, "total_inferences": total * 890, "last_sync": last_log.last_sync if last_log else datetime.utcnow()}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
