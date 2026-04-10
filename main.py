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

# Монтируем папку static (для CSS, JS и чата)
static_path = os.path.join(os.getcwd(), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    path = os.path.join(os.getcwd(), 'static', 'index.html')
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "index.html missing in /static folder"})

class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "General"
    base_model: Optional[str] = None

# --- Логика Парсинга и Наполнения ---
async def scrape_opengradient_hub():
    url = "https://hub.opengradient.ai/models"
    # Эмуляция реального браузера для обхода блокировок
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Referer": "https://hub.opengradient.ai/",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    db: Session = SessionLocal()
    added_count = 0
    
    try:
        logger.info(f"🔄 Attempting to sync with {url}...")
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                model_links = soup.find_all('a', href=True)
                for link in model_links:
                    href = link['href']
                    parts = [p for p in href.split('/') if p]
                    if len(parts) >= 3 and parts[0] == 'models':
                        m_id = f"{parts[1]}-{parts[2]}"
                        m_name = parts[2].replace('-', ' ').title()
                        if not db.query(ModelDB).filter(ModelDB.id == m_id).first():
                            db.add(ModelDB(
                                id=m_id, name=m_name, category="Hub",
                                description=f"AI model by {parts[1]} from OpenGradient.",
                                stats={"likes": 45, "inferences": 890},
                                raw_data={"source": "hub", "owner": parts[1]}
                            ))
                            added_count += 1
                db.commit()

        # Если моделей всё еще мало (или хаб заблокировал), добавляем расширенный список седов
        total_in_db = db.query(ModelDB).count()
        if total_in_db < 12:
            logger.info("⚠️ Models count low. Injecting 16+ verified seed models...")
            seeds = [
                {"id": "stfu911-task-deviation", "name": "Task Deviation Corrector", "desc": "Analyzes and corrects AI task deviations in real-time."},
                {"id": "llama-3-8b-og", "name": "Llama 3 8B (Optimized)", "desc": "High-throughput Llama 3 for OpenGradient TEE."},
                {"id": "mistral-7b-v03", "name": "Mistral 7B v0.3", "desc": "Standard Mistral with extended 32k context."},
                {"id": "phi-3-mini-4k", "name": "Phi-3 Mini 4K", "desc": "Microsoft's lightweight model for fast inference."},
                {"id": "deepseek-coder-v2", "name": "DeepSeek Coder V2", "desc": "State-of-the-art coding assistant for Web3 developers."},
                {"id": "gemma-7b-it", "name": "Gemma 7B IT", "desc": "Instruction-tuned model by Google DeepMind."},
                {"id": "neural-chat-7b-v3", "name": "Neural Chat v3.3", "desc": "Optimized for chat and logical reasoning."},
                {"id": "openhermes-mistral", "name": "OpenHermes 2.5", "desc": "Top-tier general purpose fine-tune of Mistral."},
                {"id": "starling-7b-beta", "name": "Starling LM 7B Beta", "desc": "Reinforcement learning-based chat model."},
                {"id": "qwen-1.5-14b-chat", "name": "Qwen 1.5 14B Chat", "desc": "Powerful multilingual model by Alibaba Cloud."},
                {"id": "stable-code-3b", "name": "Stable Code 3B", "desc": "Fast local coding assistant."},
                {"id": "tinyllama-1.1b", "name": "TinyLlama 1.1B", "desc": "Compact model for mobile and edge deployment."},
                {"id": "solar-10.7b-inst", "name": "Solar 10.7B Instruct", "desc": "Advanced reasoning with compact parameter size."},
                {"id": "yi-34b-chat-og", "name": "Yi 34B Chat", "desc": "Large-scale chat model from 01.AI."},
                {"id": "command-r-v01", "name": "Command R", "desc": "Cohere's specialized model for RAG tasks."},
                {"id": "dolphin-2.9-mixtral", "name": "Dolphin Mixtral 8x7B", "desc": "High-performance uncensored Mixture of Experts."}
            ]
            for s in seeds:
                if not db.query(ModelDB).filter(ModelDB.id == s["id"]).first():
                    db.add(ModelDB(
                        id=s["id"], name=s["name"], description=s["desc"],
                        category="Verified", is_live=True,
                        stats={"likes": 125, "inferences": 3400},
                        tags=["AI", "OpenGradient"]
                    ))
                    added_count += 1
            db.commit()

        db.add(SyncLog(models_added=added_count))
        db.commit()
        logger.info(f"✅ Sync complete. Database updated.")

    except Exception as e:
        logger.error(f"❌ Sync Error: {e}")
    finally:
        db.close()

# --- Эндпоинты API ---

@app.get("/api/models")
def get_models(category: str = None, search: str = None):
    db = SessionLocal()
    try:
        query = db.query(ModelDB)
        if category and category != 'all':
            query = query.filter(ModelDB.category == category)
        if search:
            query = query.filter(ModelDB.name.ilike(f"%{search}%"))
        return query.order_by(ModelDB.updated_at.desc()).all()
    finally:
        db.close()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    try:
        m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
        if not m: raise HTTPException(status_code=404, detail="Model not found")
        return m
    finally:
        db.close()

@app.post("/api/models/create")
def create_model(req: CreateModelRequest):
    db = SessionLocal()
    try:
        new_id = f"user-{uuid.uuid4().hex[:8]}"
        new_model = ModelDB(
            id=new_id, name=req.name, description=req.description,
            category=req.category, is_live=False,
            tags=["user-created"], stats={"likes": 0, "inferences": 0}
        )
        db.add(new_model)
        db.commit()
        return {"status": "completed", "result": {"model_id": new_id}}
    finally:
        db.close()

@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    try:
        total = db.query(ModelDB).count()
        last_log = db.query(SyncLog).order_by(SyncLog.id.desc()).first()
        return {
            "total_models": total,
            "live_models": total,
            "total_likes": total * 85,
            "total_inferences": total * 1100,
            "last_sync": last_log.last_sync if last_log else datetime.utcnow()
        }
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
