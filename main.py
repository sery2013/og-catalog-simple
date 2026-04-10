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
from sqlalchemy import Column, String, Boolean, Text, JSON, DateTime, create_engine, desc
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

# Модель таблицы
class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text)
    category = Column(String)
    type = Column(String, default="BASE") 
    is_live = Column(Boolean, default=True)
    stats = Column(JSON)
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# --- УПРАВЛЕНИЕ СТРУКТУРОЙ ---
# Раскомментируй drop_all для полной очистки базы при деплое
Base.metadata.drop_all(bind=engine) 
Base.metadata.create_all(bind=engine)
# -----------------------------

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
    base_model: str # Выбор базового движка

# Эндпоинты
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

@app.get("/api/models")
def get_models():
    db = SessionLocal()
    try:
        # Сортировка: новые сверху
        return db.query(ModelDB).order_by(desc(ModelDB.created_at)).all()
    finally:
        db.close()

@app.post("/api/chat")
async def chat(req: ChatRequest):
    tech_responses = [
        f"Inference successful on TEE node #0x{uuid.uuid4().hex[:4]}. Latency: {random.randint(20, 95)}ms.",
        f"Verified via OpenGradient SDK. Integrity score: 0.999. Hash: 0x{uuid.uuid4().hex[:10]}",
        f"Model {req.model_id} optimized. Resource usage: {random.uniform(0.1, 0.3):.3f} tokens.",
        f"Consensus reached. Secure enclave status: ACTIVE."
    ]
    return {"reply": random.choice(tech_responses)}

@app.post("/api/models/create")
def create_model(req: CreateModelRequest):
    db = SessionLocal()
    try:
        # ID теперь включает имя базовой модели
        new_id = f"{req.base_model.lower()}-{uuid.uuid4().hex[:4]}"
        new_model = ModelDB(
            id=new_id,
            name=req.name,
            description=req.description,
            category="AI Model",
            type="USER", 
            is_live=True,
            stats={"likes": 0, "inferences": 0},
            raw_data={
                "base_engine": req.base_model,
                "status": "deployed",
                "sdk_version": "0.4.2",
                "deployment_hash": uuid.uuid4().hex
            },
            created_at=datetime.utcnow() 
        )
        db.add(new_model)
        db.commit()
        return {"status": "success", "id": new_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

async def init_db():
    db = SessionLocal()
    try:
        if db.query(ModelDB).count() == 0:
            seeds = [
                {"id": "llama-3-8b", "name": "Llama 3 8B Gradient", "desc": "Meta's latest model optimized by Gradient.", "type": "BASE"},
                {"id": "mistral-7b", "name": "Mistral 7B v0.3", "desc": "High-performance compact NLP model.", "type": "LIVE"},
                {"id": "phi-3-mini", "name": "Phi-3 Mini 4K", "desc": "Microsoft lightweight small language model.", "type": "BASE"}
            ]
            for s in seeds:
                db.add(ModelDB(
                    id=s["id"], name=s["name"], description=s["desc"],
                    category="General", type=s["type"],
                    stats={"likes": random.randint(100, 500), "inferences": random.randint(1000, 9000)},
                    raw_data={"provider": "OpenGradient", "tier": "Verified"},
                    created_at=datetime.utcnow()
                ))
            db.commit()
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    await init_db()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
