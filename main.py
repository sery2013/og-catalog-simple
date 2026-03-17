from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
import httpx
import os
import json
import asyncio
import uuid

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="4.1.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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

# === 12 БАЗОВЫХ МОДЕЛЕЙ ===
BASE_MODELS = [
    ModelInfo(id="og-1hr-volatility-ethusdt", name="ETH/USDT 1hr Volatility", description="Real-time ETH/USDT volatility forecasting model", category="Risk", tags=["defi", "prediction", "ethereum"], stats={"likes": 42, "inferences": 1287}),
    ModelInfo(id="og-llama3-fintune-v2", name="Llama3 Financial Fine-tuned", description="Fine-tuned Llama 3 8B for financial analysis", category="Language", tags=["llm", "nlp", "finance"], stats={"likes": 156, "inferences": 5432}),
    ModelInfo(id="og-risk-bert-base", name="Risk Assessment BERT", description="BERT-based model for DeFi risk scoring", category="Risk", tags=["risk", "defi", "security"], stats={"likes": 89, "inferences": 3421}),
    ModelInfo(id="og-defi-gemma", name="DeFi Gemma Assistant", description="Gemma 7B specialized in DeFi protocols", category="DeFi", tags=["defi", "llm", "yield"], stats={"likes": 203, "inferences": 8765}),
    ModelInfo(id="og-amm-optimizer", name="AMM Fee Optimizer", description="Optimizes trading fees for AMMs", category="Protocol", tags=["defi", "amm", "trading"], stats={"likes": 67, "inferences": 2100}),
    ModelInfo(id="og-sybil-detector", name="Sybil Detection Model", description="GNN for detecting sybil attacks", category="Risk", tags=["security", "defi", "gnn"], stats={"likes": 134, "inferences": 4567}),
    ModelInfo(id="og-sentiment-crypto", name="Crypto Sentiment Analyzer", description="Real-time sentiment analysis for crypto", category="Language", tags=["sentiment", "nlp", "crypto"], stats={"likes": 178, "inferences": 6543}),
    ModelInfo(id="og-liquidation-predictor", name="Liquidation Predictor", description="Predicts liquidation events in lending", category="Risk", tags=["defi", "lending", "prediction"], stats={"likes": 95, "inferences": 3210}),
    ModelInfo(id="og-yield-optimizer", name="Yield Farming Optimizer", description="Optimizes yield farming strategies", category="DeFi", tags=["defi", "yield", "farming"], stats={"likes": 221, "inferences": 7890}),
    ModelInfo(id="og-nft-pricer", name="NFT Price Predictor", description="Predicts NFT floor prices", category="Multimodal", tags=["nft", "prediction", "pricing"], stats={"likes": 112, "inferences": 2890}),
    ModelInfo(id="og-mev-detector", name="MEV Opportunity Detector", description="Detects MEV opportunities in real-time", category="Protocol", tags=["mev", "trading", "arbitrage"], stats={"likes": 187, "inferences": 5670}),
    ModelInfo(id="og-portfolio-advisor", name="DeFi Portfolio Advisor", description="AI advisor for DeFi portfolio", category="Language", tags=["defi", "portfolio", "advisor"], stats={"likes": 156, "inferences": 4320}),
]

# === БАЗА ДАННЫХ (исправлено для SQLAlchemy 2.0) ===
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL")
engine = None
SessionLocal = None
DBModel = None
db_ok = False

try:
    if DATABASE_URL:
        from sqlalchemy import create_engine, Column, String, JSON, DateTime, Boolean
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
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
            synced_at = Column(DateTime, nullable=True)
        
        Base.metadata.create_all(bind=engine)
        
        # ✅ Тест подключения (SQLAlchemy 2.0 syntax)
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        db_ok = True
        logger.info("✅ Database connected and verified")
    else:
        logger.warning("⚠️ No DATABASE_URL env var")
except Exception as e:
    logger.warning(f"⚠️ Database connection test failed: {e}")
    # 🔍 Fallback: проверяем на практике
    try:
        if DATABASE_URL and SessionLocal:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            db_ok = True
            logger.info("✅ Database works (fallback check passed)")
        else:
            db_ok = False
            logger.info("📦 Running in MEMORY-ONLY mode")
    except:
        db_ok = False
        logger.info("📦 Running in MEMORY-ONLY mode")

# === ХРАНИЛИЩА ===
chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}
memory_models: Dict[str, dict] = {}
sync_status = {"last_sync": None, "models_added": 0, "errors": []}

# === ФУНКЦИИ ===
def get_all_models():
    """Возвращает все модели"""
    result = []
    
    # Из БД
    if db_ok and SessionLocal:
        try:
            db = SessionLocal()
            for m in db.query(DBModel).filter(DBModel.is_live == True).all():
                result.append(ModelInfo(
                    id=m.id, name=m.name, description=m.description, category=m.category,
                    tags=m.tags if isinstance(m.tags, list) else [],
                    stats=m.stats if isinstance(m.stats, dict) else {"likes":0,"inferences":0},
                    created_at=m.created_at, is_live=True
                ))
            for m in db.query(DBModel).filter(DBModel.is_user_created == True).all():
                result.append(ModelInfo(
                    id=m.id, name=m.name, description=m.description, category=m.category,
                    tags=m.tags if isinstance(m.tags, list) else [],
                    stats=m.stats if isinstance(m.stats, dict) else {"likes":0,"inferences":0},
                    created_at=m.created_at, is_live=False
                ))
            db.close()
        except Exception as e:
            logger.warning(f"⚠️ DB read error: {e}")
    
    # Из памяти
    for mdata in memory_models.values():
        result.append(ModelInfo(**mdata))
    
    # Базовые
    result += BASE_MODELS
    return result

async def fetch_live_models():
    """Получает live модели"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://hub.opengradient.ai/api/models")
            if resp.status_code == 200:
                data = resp.json()
                items = data if isinstance(data, list) else data.get('data', [])
                models = []
                for item in items[:5]:
                    repo = item.get('model_repository', item.get('repository', item))
                    if isinstance(repo, dict) and repo.get('description'):
                        models.append({
                            "id": repo.get('name', f"og-{len(models)}"),
                            "name": repo.get('name', 'Unknown'),
                            "description": repo.get('description', ''),
                            "category": repo.get('category', 'Uncategorized'),
                            "tags": repo.get('tags', []),
                            "stats": {"likes": 0, "inferences": 0},
                            "created_at": datetime.now().isoformat(),
                            "is_live": True
                        })
                return models
    except Exception as e:
        logger.warning(f"⚠️ Fetch error: {e}")
    return []

async def sync_task():
    """Синхронизация"""
    logger.info("🔄 Syncing...")
    try:
        sync_status["last_sync"] = datetime.now().isoformat()
        if not db_ok or not SessionLocal: 
            logger.warning("⚠️ DB not available for sync")
            return
        
        live = await fetch_live_models()
        if not live: return
        
        db = SessionLocal()
        added = 0
        for lm in live:
            if not db.query(DBModel).filter(DBModel.id == lm["id"]).first():
                db.add(DBModel(
                    id=lm["id"], name=lm["name"], description=lm["description"],
                    category=lm["category"], tags=json.dumps(lm["tags"]),
                    stats=json.dumps(lm["stats"]), created_at=lm.get("created_at"),
                    is_live=True, is_user_created=False, synced_at=datetime.now()
                ))
                added += 1
                if added >= 5: break
        db.commit()
        db.close()
        sync_status["models_added"] = added
        logger.info(f"✅ Synced +{added}")
    except Exception as e:
        logger.error(f"❌ Sync error: {e}")

# === СОЗДАНИЕ МОДЕЛИ ===
async def create_model_task(task_id: str, req: CreateModelRequest):
    try:
        model_tasks[task_id]["status"] = "processing"
        for p in [25, 50, 75, 100]:
            model_tasks[task_id]["progress"] = p
            await asyncio.sleep(0.3)
        
        model_id = f"og-{req.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
        model_data = {
            "id": model_id, "name": req.name, "description": req.description,
            "category": req.category, "tags": ["custom", "user-created"],
            "stats": {"likes": 0, "inferences": 0},
            "created_at": datetime.now().isoformat(), "is_live": False
        }
        
        # Сохраняем в БД или память
        if db_ok and SessionLocal:
            try:
                db = SessionLocal()
                db.add(DBModel(
                    id=model_id, name=req.name, description=req.description,
                    category=req.category, tags=json.dumps(model_data["tags"]),
                    stats=json.dumps(model_data["stats"]),
                    created_at=model_data["created_at"],
                    is_live=False, is_user_created=True, synced_at=datetime.now()
                ))
                db.commit()
                db.close()
                logger.info(f"✅ Saved to DB: {model_id}")
            except Exception as e:
                logger.warning(f"⚠️ DB save failed: {e}, saving to memory")
                memory_models[model_id] = model_data
        else:
            memory_models[model_id] = model_data
            logger.info(f"💾 Saved to memory: {model_id}")
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Created",
            "tx_hash": "0x" + os.urandom(32).hex()
        }
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)
        logger.error(f"❌ Task error: {e}")

# === CHAT ===
async def chat_response(query: str, model: Optional[ModelInfo] = None) -> str:
    if model:
        return f"**{model.name}** ({model.category})\n\n{model.description}\n\nTags: {', '.join(model.tags)}\n\nStats: {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences"
    return "👋 Select a model to chat!"

# === ROUTES ===
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "4.1.0",
        "total_models": len(get_all_models()),
        "database": "✓" if db_ok else "✗ (memory mode)",
        "sync_status": sync_status
    }

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None, live_only: bool = False):
    models = get_all_models()
    if live_only: models = [m for m in models if m.is_live]
    if category and category != 'all': models = [m for m in models if m.category.lower() == category.lower()]
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower()]
    return [m.model_dump() for m in models[:100]]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    for m in get_all_models():
        if m.id == model_id: return m.model_dump()
    raise HTTPException(404, "Not found")

@app.get("/api/stats")
async def get_stats():
    models = get_all_models()
    return {
        "total_models": len(models),
        "live_models": len([m for m in models if m.is_live]),
        "total_likes": sum(m.stats.get('likes', 0) for m in models),
        "total_inferences": sum(m.stats.get('inferences', 0) for m in models),
        "last_sync": sync_status.get("last_sync"),
        "sync_added": sync_status.get("models_added", 0)
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model = next((m for m in get_all_models() if m.id == req.model_id), None) if req.model_id else None
    reply = await chat_response(req.query, model)
    if sid not in chat_sessions: chat_sessions[sid] = []
    chat_sessions[sid].append({"role": "user", "content": req.query})
    chat_sessions[sid].append({"role": "assistant", "content": reply})
    return {"reply": reply, "session_id": sid}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status": "queued", "progress": 0}
    bg.add_task(create_model_task, tid, req)
    return {"task_id": tid, "status": "queued"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t: raise HTTPException(404, "Not found")
    return t

@app.on_event("startup")
async def startup():
    logger.info("🚀 OpenGradient Catalog starting...")
    asyncio.create_task(asyncio.sleep(5)).add_done_callback(lambda _: asyncio.create_task(sync_task()))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
