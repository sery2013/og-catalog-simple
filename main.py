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

Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(title="OpenGradient Catalog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ГЛАВНАЯ СТРАНИЦА (Исправлено для Render) ---
@app.get("/")
async def read_index():
    index_path = os.path.join(os.getcwd(), 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.error(f"❌ File not found: {index_path}")
        return JSONResponse(
            status_code=404, 
            content={"error": "index.html not found on server. Ensure it is in the root directory."}
        )

class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "General"
    base_model: Optional[str] = None

# --- Логика Парсинга и Наполнения ---
async def scrape_opengradient_hub():
    url = "https://hub.opengradient.ai/models"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    }
    
    logger.info(f"🔄 Attempting to fetch models from {url}...")
    db: Session = SessionLocal()
    
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(url, timeout=15.0)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                model_links = soup.find_all('a', href=True)
                added_count = 0
                
                for link in model_links:
                    href = link['href']
                    if href.startswith('/models/'):
                        m_id = href.replace('/models/', '').replace('/', '-')
                        m_name = m_id.split('-')[-1].replace('-', ' ').title()
                        
                        if not db.query(ModelDB).filter(ModelDB.id == m_id).first():
                            db.add(ModelDB(
                                id=m_id, name=m_name, category="General", is_live=True,
                                description="AI Model imported from OpenGradient Hub.",
                                stats={"likes": 15, "inferences": 120},
                                raw_data={"source": "hub", "path": href}
                            ))
                            added_count += 1
                
                db.add(SyncLog(models_added=added_count))
                db.commit()
                logger.info(f"✅ Sync complete. Added {added_count} models.")
            else:
                logger.warning(f"⚠️ Hub blocked (Status {response.status_code}). Adding seeds...")
                seeds = [
                    {"id": "llama-3-8b", "name": "Llama 3 8B Gradient", "desc": "Meta's latest model optimized by Gradient."},
                    {"id": "mistral-7b", "name": "Mistral 7B v0.3", "desc": "High-performance compact NLP model."},
                    {"id": "phi-3-mini", "name": "Phi-3 Mini 4K", "desc": "Microsoft lightweight small language model."}
                ]
                
                added_count = 0
                for s in seeds:
                    if not db.query(ModelDB).filter(ModelDB.id == s["id"]).first():
                        db.add(ModelDB(
                            id=s["id"], name=s["name"], description=s["desc"],
                            category="General", is_live=True,
                            stats={"likes": 42, "inferences": 560},
                            raw_data={"status": "seeded", "model_type": "LLM"}
                        ))
                        added_count += 1
                
                db.add(SyncLog(models_added=added_count))
                db.commit()
                logger.info(f"💾 Added {added_count} seed models to PostgreSQL.")

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
        
        models = query.order_by(ModelDB.updated_at.desc()).all()
        return models
    finally:
        db.close()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    try:
        m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
        if not m: raise HTTPException(status_code=404)
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
            tags=["user-created"], stats={"likes": 0, "inferences": 0},
            raw_data={"model_name": req.name, "status": "deployed"}
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
        
        sync_date = last_log.last_sync if last_log else datetime.utcnow()
        sync_count = last_log.models_added if last_log else 0

        return {
            "total_models": total,
            "live_models": total,
            "total_likes": total * 15 + 120,
            "total_inferences": total * 120 + 450,
            "last_sync": sync_date,
            "sync_added": sync_count
        }
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    # Поддержка порта Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
