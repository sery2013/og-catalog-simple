import os
import asyncio
import logging
import uuid
from typing import List, Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# ИСПРАВЛЕНО: удален create_all из импорта, так как он не нужен здесь
from sqlalchemy import Column, String, Integer, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from bs4 import BeautifulSoup

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Настройка БД ---
DATABASE_URL = os.getenv("DATABASE_URL")

# Исправление протокола для SQLAlchemy 1.4+
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    logger.warning("⚠️ DATABASE_URL not found, using SQLite test.db")
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

# Создаем таблицы (используем встроенный метод метаданных)
Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="OpenGradient Catalog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Схемы Pydantic ---
class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "General"
    base_model: Optional[str] = None

# --- Логика Парсинга (Scraping) ---
async def scrape_opengradient_hub():
    url = "https://hub.opengradient.ai/models"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    logger.info(f"🔄 Starting HTML scrape from {url}...")
    
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            response = await client.get(url, timeout=15.0)
            if response.status_code != 200:
                logger.error(f"❌ Hub returned status {response.status_code}")
                return

            soup = BeautifulSoup(response.text, 'html.parser')
            model_links = soup.find_all('a', href=True)
            added_count = 0
            
            db: Session = SessionLocal()
            for link in model_links:
                href = link['href']
                if href.startswith('/models/'):
                    m_id = href.replace('/models/', '')
                    # Формируем красивое имя из ID
                    m_name = m_id.split('/')[-1].replace('-', ' ').title()
                    
                    existing = db.query(ModelDB).filter(ModelDB.id == m_id).first()
                    if not existing:
                        new_model = ModelDB(
                            id=m_id,
                            name=m_name,
                            description="AI Model imported from OpenGradient Hub.",
                            category="General",
                            is_live=True,
                            tags=["automated-import", "hub"],
                            stats={"likes": 12, "inferences": 140},
                            raw_data={"source": "hub", "path": href} # Данные для копирования конфига
                        )
                        db.add(new_model)
                        added_count += 1
            
            db.add(SyncLog(models_added=added_count))
            db.commit()
            db.close()
            logger.info(f"✅ Sync complete. Added {added_count} new models.")
            
        except Exception as e:
            logger.error(f"❌ Scrape error: {e}")

# --- Эндпоинты API ---

@app.get("/api/models")
def get_models(category: str = None, search: str = None):
    db = SessionLocal()
    query = db.query(ModelDB)
    if category and category != 'all':
        query = query.filter(ModelDB.category == category)
    if search:
        query = query.filter(ModelDB.name.ilike(f"%{search}%"))
    
    models = query.order_by(ModelDB.updated_at.desc()).all()
    db.close()
    return models

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    db.close()
    if not m:
        raise HTTPException(status_code=404, detail="Model not found")
    return m

@app.post("/api/models/create")
def create_model(req: CreateModelRequest):
    db = SessionLocal()
    new_id = f"user-{uuid.uuid4().hex[:8]}"
    new_model = ModelDB(
        id=new_id,
        name=req.name,
        description=req.description,
        category=req.category,
        is_live=False,
        tags=["user-created"],
        stats={"likes": 0, "inferences": 0},
        raw_data={
            "model_name": req.name,
            "base": req.base_model or "unknown",
            "status": "deployed"
        }
    )
    db.add(new_model)
    db.commit()
    db.close()
    return {"status": "completed", "result": {"model_id": new_id}}

@app.get("/api/stats")
def get_stats():
    db = SessionLocal()
    total = db.query(ModelDB).count()
    live = db.query(ModelDB).filter(ModelDB.is_live == True).count()
    last_log = db.query(SyncLog).order_by(SyncLog.id.desc()).first()
    db.close()
    return {
        "total_models": total,
        "live_models": live,
        "total_likes": total * 15,
        "total_inferences": total * 120,
        "last_sync": last_log.last_sync if last_log else None,
        "sync_added": last_log.models_added if last_log else 0
    }

# --- Запуск задач ---
@app.on_event("startup")
async def startup_event():
    # Запуск парсинга в фоновом потоке
    asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
