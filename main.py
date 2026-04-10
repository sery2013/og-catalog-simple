import os
import asyncio
import logging
import uuid
import random
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, String, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка базы данных
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Модель таблицы в БД
class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text)
    category = Column(String)
    is_live = Column(Boolean, default=True)
    stats = Column(JSON)
    raw_data = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Подключение статики
static_path = os.path.join(os.getcwd(), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Схемы данных
class ChatRequest(BaseModel):
    model_id: str
    message: str

class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "Custom"

# Маршруты
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

@app.get("/api/models")
def get_models():
    db = SessionLocal()
    try:
        return db.query(ModelDB).all()
    finally:
        db.close()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    try:
        m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
        if not m:
            raise HTTPException(status_code=404, detail="Model not found")
        return m
    finally:
        db.close()

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Разнообразные ответы в стиле OpenGradient SDK
    tech_responses = [
        f"Inference successful on TEE node #0x{uuid.uuid4().hex[:4]}. Latency: {random.randint(15, 85)}ms.",
        f"Verified via OpenGradient SDK. Integrity score: 0.999. Transaction hash: 0x{uuid.uuid4().hex[:12]}...",
        f"Model {req.model_id} processed task. Consensus reached across 7 validator nodes.",
        f"Secure enclave active. Resource consumption: {random.uniform(0.05, 0.25):.4f} OG tokens.",
        f"Analysis complete via OpenGradient SDK. Output verified for bias and variance."
    ]
    return {"reply": random.choice(tech_responses)}

@app.post("/api/models/create")
def create_fork(req: CreateModelRequest):
    db = SessionLocal()
    try:
        new_id = f"fork-{uuid.uuid4().hex[:6]}"
        new_m = ModelDB(
            id=new_id,
            name=req.name,
            description=req.description,
            category=req.category,
            is_live=True,
            stats={"likes": 0, "inferences": 0},
            raw_data={"status": "forked", "base": "llama-3"}
        )
        db.add(new_m)
        db.commit()
        return {"result": {"model_id": new_id}}
    finally:
        db.close()

# Начальное заполнение базы (Seeding)
async def init_db():
    db = SessionLocal()
    try:
        if db.query(ModelDB).count() == 0:
            seeds = [
                {
                    "id": "llama-3-8b", 
                    "name": "Llama 3 8B Gradient", 
                    "desc": "Meta's latest model optimized by Gradient.", 
                    "cat": "LLM",
                    "raw": {"model_type": "LLM", "provider": "Meta", "tier": "Secure"}
                },
                {
                    "id": "mistral-7b", 
                    "name": "Mistral 7B v0.3", 
                    "desc": "High-performance compact NLP model.", 
                    "cat": "General",
                    "raw": {"model_type": "NLP", "provider": "Mistral", "tier": "Standard"}
                },
                {
                    "id": "stfu911-corrector", 
                    "name": "Task Deviation Corrector", 
                    "desc": "Real-time correction for AI agent output variance.", 
                    "cat": "Agentic",
                    "raw": {"model_type": "Utility", "provider": "OpenGradient", "tier": "Pro"}
                }
            ]
            for s in seeds:
                db.add(ModelDB(
                    id=s["id"], 
                    name=s["name"], 
                    description=s["desc"], 
                    category=s["cat"],
                    stats={"likes": random.randint(10, 50), "inferences": random.randint(100, 1000)},
                    raw_data=s["raw"]
                ))
            db.commit()
            logger.info("Database seeded with models.")
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    await init_db()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
