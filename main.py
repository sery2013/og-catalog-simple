import os
import asyncio
import logging
import uuid
from typing import List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles # Добавлено для статики
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
    is_live = Column(Boolean, default=False)
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

# Подключаем папку static, чтобы работали стили и скрипты из нее
if os.path.exists(os.path.join(os.getcwd(), 'static')):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# --- ГЛАВНАЯ СТРАНИЦА (Теперь точно найдет!) ---
@app.get("/")
async def read_index():
    current_dir = os.getcwd()
    # Твой файл нашелся в папке static, поэтому ставим её на первое место
    possible_paths = [
        os.path.join(current_dir, 'static', 'index.html'),
        os.path.join(current_dir, 'index.html'),
        os.path.join(current_dir, 'frontend', 'index.html'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found index.html at: {path}")
            return FileResponse(path)
    
    return JSONResponse(status_code=404, content={"error": "index.html not found"})

# --- Остальные эндпоинты остаются без изменений ---

class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "General"

async def scrape_opengradient_hub():
    url = "https://hub.opengradient.ai/models"
    headers = {"User-Agent": "Mozilla/5.0"}
    db = SessionLocal()
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                # (Логика парсинга...)
                pass 
            else:
                # Добавление сидов если хаб недоступен
                seeds = [{"id": "llama-3", "name": "Llama 3 Gradient", "desc": "Seeded model"}]
                for s in seeds:
                    if not db.query(ModelDB).filter(ModelDB.id == s["id"]).first():
                        db.add(ModelDB(id=s["id"], name=s["name"], description=s["desc"], category="General"))
                db.commit()
    except Exception as e:
        logger.error(f"Sync Error: {e}")
    finally:
        db.close()

@app.get("/api/models")
def get_models():
    db = SessionLocal()
    try:
        return db.query(ModelDB).all()
    finally:
        db.close()

@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    try:
        total = db.query(ModelDB).count()
        return {"total_models": total, "last_sync": datetime.utcnow()}
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
