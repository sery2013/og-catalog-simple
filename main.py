from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import httpx
import os
import json
import asyncio
import uuid

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="4.2.0")

# Разрешаем CORS для всех доменов
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# === МОДЕЛИ ДАННЫХ ===
class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    name: str
    description: str
    category: str
    tags: List[str] = []
    author: str = "OpenGradient"
    stats: Dict[str, int] = {"likes": 0, "inferences": 0}
    created_at: Optional[str] = None
    is_live: bool = False
    raw_data: Optional[Dict[str, Any]] = None # Для копирования JSON кода

class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    query: str
    model_id: Optional[str] = None
    session_id: Optional[str] = None

class CreateModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    description: str
    category: str
    base_model: Optional[str] = None

# === БАЗА ДАННЫХ ===
DATABASE_URL = os.getenv("DATABASE_URL")
SessionLocal = None
DBModel = None

if DATABASE_URL:
    try:
        from sqlalchemy import create_engine, Column, String, JSON, DateTime, Boolean
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        # Настройки для Render Postgres (Internal URL)
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine)
        Base = declarative_base()
        
        class DBModel(Base):
            __tablename__ = "models"
            id = Column(String, primary_key=True, index=True)
            name = Column(String, nullable=False)
            description = Column(String, nullable=False)
            category = Column(String, nullable=False)
            tags = Column(JSON, nullable=True)
            stats = Column(JSON, nullable=True)
            created_at = Column(String, nullable=True)
            is_live = Column(Boolean, default=False)
            is_user_created = Column(Boolean, default=False)
            raw_data = Column(JSON, nullable=True) # Поле для хранения полного конфига

        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ DB init error: {e}. Switching to memory mode.")

# === ХРАНИЛИЩА (ДЛЯ MEMORY MODE) ===
memory_models: Dict[str, dict] = {}
sync_status = {"last_sync": None, "models_added": 0}

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def get_all_models():
    result = []
    if SessionLocal:
        try:
            db = SessionLocal()
            models = db.query(DBModel).all()
            for m in models:
                # Безопасно парсим JSON если он пришел строкой
                tags = m.tags if isinstance(m.tags, list) else json.loads(m.tags or "[]")
                stats = m.stats if isinstance(m.stats, dict) else json.loads(m.stats or '{"likes":0,"inferences":0}')
                raw = m.raw_data if isinstance(m.raw_data, dict) else json.loads(m.raw_data or "{}")
                
                result.append(ModelInfo(
                    id=m.id, name=m.name, description=m.description, category=m.category,
                    tags=tags, stats=stats, created_at=m.created_at, 
                    is_live=m.is_live, raw_data=raw
                ))
            db.close()
        except Exception as e:
            logger.error(f"DB Error: {e}")
    
    # Добавляем модели из памяти (если БД упала или для новых сессий)
    for mid in memory_models:
        if not any(r.id == mid for r in result):
            result.append(ModelInfo(**memory_models[mid]))
            
    return result

async def fetch_live_models():
    """Спарсить модели с защитой от 403 ошибки"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get("https://hub.opengradient.ai/api/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                items = data if isinstance(data, list) else data.get('data', [])
                return items[:15] # Берем первые 15 для каталога
    except Exception as e:
        logger.error(f"Scraper error: {e}")
    return []

async def sync_task():
    """Фоновая синхронизация"""
    logger.info("🔄 Syncing with OpenGradient Hub...")
    live_data = await fetch_live_models()
    if not live_data or not SessionLocal: return

    db = SessionLocal()
    added = 0
    try:
        for item in live_data:
            # Пытаемся вытащить имя и описание из разных структур API
            repo = item.get('model_repository', item)
            m_id = repo.get('name', str(uuid.uuid4())[:8])
            
            if not db.query(DBModel).filter(DBModel.id == m_id).first():
                new_m = DBModel(
                    id=m_id,
                    name=repo.get('name', 'AI Model'),
                    description=repo.get('description', 'No description available'),
                    category=repo.get('category', 'General'),
                    tags=repo.get('tags', []),
                    stats={"likes": 0, "inferences": 0},
                    created_at=datetime.now().isoformat(),
                    is_live=True,
                    raw_data=item # Сохраняем весь объект для копирования кода
                )
                db.add(new_m)
                added += 1
        db.commit()
        sync_status["last_sync"] = datetime.now().isoformat()
        sync_status["models_added"] = added
        logger.info(f"✅ Sync complete. Added {added} models.")
    except Exception as e:
        logger.error(f"Sync commit error: {e}")
    finally:
        db.close()

# === ROUTES ===
@app.get("/")
async def root():
    # Отдаем index.html из корня или static
    for path in ["static/index.html", "index.html"]:
        if os.path.exists(path): return FileResponse(path)
    return {"error": "index.html not found"}

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None):
    models = get_all_models()
    if category and category != 'all':
        models = [m for m in models if m.category.lower() == category.lower()]
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower()]
    return models

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    models = get_all_models()
    model = next((m for m in models if m.id == model_id), None)
    if not model: raise HTTPException(404, "Model not found")
    return model

@app.get("/api/stats")
async def get_stats():
    models = get_all_models()
    return {
        "total_models": len(models),
        "live_models": len([m for m in models if m.is_live]),
        "total_likes": sum(m.stats.get('likes', 0) for m in models),
        "total_inferences": sum(m.stats.get('inferences', 0) for m in models),
        "last_sync": sync_status["last_sync"]
    }

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest):
    m_id = f"custom-{uuid.uuid4().hex[:6]}"
    new_model = {
        "id": m_id, "name": req.name, "description": req.description,
        "category": req.category, "tags": ["user-created"],
        "stats": {"likes": 0, "inferences": 0}, "is_live": False,
        "raw_data": {"info": "User created model", "base": req.base_model}
    }
    
    if SessionLocal:
        db = SessionLocal()
        db.add(DBModel(
            id=m_id, name=req.name, description=req.description,
            category=req.category, tags=new_model["tags"], stats=new_model["stats"],
            is_live=False, is_user_created=True, raw_data=new_model["raw_data"]
        ))
        db.commit()
        db.close()
    else:
        memory_models[m_id] = new_model
        
    return {"status": "completed", "result": {"model_id": m_id}}

@app.on_event("startup")
async def startup():
    # Запуск синхронизации через 5 секунд после старта
    asyncio.create_task(asyncio.sleep(5)).add_done_callback(lambda _: asyncio.create_task(sync_task()))

if __name__ == "__main__":
    import uvicorn
    # ПОРТ ДЛЯ RENDER (читаем из среды)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
