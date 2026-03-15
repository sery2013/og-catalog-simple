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

# === БАЗА ДАННЫХ ===
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# === ПЛАНИРОВЩИК ===
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="4.0.0-LIVE", docs_url="/docs")
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
    is_live: bool = False  # 🔥 Новая модель с хаб-а

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
    wallet_key: Optional[str] = None

# === 12 БАЗОВЫХ МОДЕЛЕЙ (FALLBACK) ===

BASE_MODELS = [
    ModelInfo(id="og-1hr-volatility-ethusdt", name="ETH/USDT 1hr Volatility", description="Real-time ETH/USDT volatility forecasting model trained on 1-hour OHLCV data using GARCH architecture", category="Risk", tags=["defi", "prediction", "timeseries", "ethereum", "volatility"], stats={"likes": 42, "inferences": 1287}),
    ModelInfo(id="og-llama3-fintune-v2", name="Llama3 Financial Fine-tuned", description="Fine-tuned Llama 3 8B model for financial analysis, sentiment detection, and market prediction", category="Language", tags=["llm", "nlp", "finance", "sentiment", "llama3"], stats={"likes": 156, "inferences": 5432}),
    ModelInfo(id="og-risk-bert-base", name="Risk Assessment BERT", description="BERT-based model for DeFi risk scoring, sybil detection, and protocol safety analysis", category="Risk", tags=["risk", "defi", "classification", "security", "bert"], stats={"likes": 89, "inferences": 3421}),
    ModelInfo(id="og-defi-gemma", name="DeFi Gemma Assistant", description="Gemma 7B model specialized in DeFi protocols, yield farming strategies, and liquidity optimization", category="DeFi", tags=["defi", "llm", "yield", "protocols", "gemma"], stats={"likes": 203, "inferences": 8765}),
    ModelInfo(id="og-amm-optimizer", name="AMM Fee Optimizer", description="Optimizes trading fees for automated market makers based on volatility and volume", category="Protocol", tags=["defi", "amm", "optimization", "trading", "fees"], stats={"likes": 67, "inferences": 2100}),
    ModelInfo(id="og-sybil-detector", name="Sybil Detection Model", description="Graph neural network for detecting sybil attacks in DeFi protocols and identifying malicious actors", category="Risk", tags=["security", "defi", "gnn", "sybil", "detection"], stats={"likes": 134, "inferences": 4567}),
    ModelInfo(id="og-sentiment-crypto", name="Crypto Sentiment Analyzer", description="Real-time sentiment analysis for crypto markets from social media, news, and on-chain data", category="Language", tags=["sentiment", "nlp", "crypto", "social", "analysis"], stats={"likes": 178, "inferences": 6543}),
    ModelInfo(id="og-liquidation-predictor", name="Liquidation Predictor", description="Predicts liquidation events in lending protocols 30 minutes in advance using ML", category="Risk", tags=["defi", "lending", "prediction", "risk", "liquidation"], stats={"likes": 95, "inferences": 3210}),
    ModelInfo(id="og-yield-optimizer", name="Yield Farming Optimizer", description="Optimizes yield farming strategies across multiple DeFi protocols for maximum APY", category="DeFi", tags=["defi", "yield", "optimization", "farming", "strategy"], stats={"likes": 221, "inferences": 7890}),
    ModelInfo(id="og-nft-pricer", name="NFT Price Predictor", description="Predicts NFT floor prices using on-chain data, market trends, and collection metrics", category="Multimodal", tags=["nft", "prediction", "pricing", "marketplace", "ai"], stats={"likes": 112, "inferences": 2890}),
    ModelInfo(id="og-mev-detector", name="MEV Opportunity Detector", description="Detects MEV opportunities in real-time from mempool data and transaction patterns", category="Protocol", tags=["mev", "trading", "arbitrage", "defi", "mempool"], stats={"likes": 187, "inferences": 5670}),
    ModelInfo(id="og-portfolio-advisor", name="DeFi Portfolio Advisor", description="AI advisor for DeFi portfolio optimization, risk management, and asset allocation", category="Language", tags=["defi", "portfolio", "advisor", "optimization", "risk"], stats={"likes": 156, "inferences": 4320}),
]

# === БАЗА ДАННЫХ ===

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(bind=engine)
        Base = declarative_base()
        
        class DBModel(Base):
            __tablename__ = "models"
            id = Column(String, primary_key=True)
            name = Column(String)
            description = Column(String)
            category = Column(String)
            tags = Column(JSON)
            stats = Column(JSON)
            created_at = Column(String)
            is_live = Column(Boolean, default=False)  # 🔥 С хаб-а или нет
            is_user_created = Column(Boolean, default=False)  # Создана пользователем
            synced_at = Column(DateTime, nullable=True)  # Когда синхронизирована
        
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        DBModel = None
        SessionLocal = None
else:
    logger.warning("⚠️ No DATABASE_URL - using memory only")
    DBModel = None
    SessionLocal = None

# === ХРАНИЛИЩА ===

chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}
sync_status = {"last_sync": None, "models_added": 0, "errors": []}

# === ПЛАНИРОВЩИК ===

scheduler = AsyncIOScheduler()

# === ФУНКЦИИ ===

def get_all_models():
    """🔥 Возвращает: Live-модели (новые) → Пользовательские → Базовые (фоллбэк)"""
    if DBModel and SessionLocal:
        try:
            db = SessionLocal()
            # 🔥 Сортировка: live-модели первыми, потом пользовательские, потом базовые
            live_models = db.query(DBModel).filter(
                DBModel.is_live == True
            ).order_by(DBModel.synced_at.desc()).all()
            
            user_models = db.query(DBModel).filter(
                DBModel.is_user_created == True
            ).all()
            
            def to_model_info(m):
                return ModelInfo(
                    id=m.id, name=m.name, description=m.description, category=m.category,
                    tags=m.tags if isinstance(m.tags, list) else (json.loads(m.tags) if m.tags else []),
                    stats=m.stats if isinstance(m.stats, dict) else (json.loads(m.stats) if m.stats else {"likes":0,"inferences":0}),
                    created_at=m.created_at,
                    is_live=m.is_live
                )
            
            result = [to_model_info(m) for m in live_models]
            result += [to_model_info(m) for m in user_models]
            result += BASE_MODELS  # Фоллбэк
            db.close()
            return result
        except Exception as e:
            logger.error(f"DB read error: {e}")
            return BASE_MODELS
    return BASE_MODELS

async def fetch_live_models_from_hub() -> List[Dict]:
    """🔥 Получает модели с официального хаба"""
    live_models = []
    
    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OpenGradient-Catalog/4.0"}) as client:
            # Пробуем разные эндпоинты (официальный хаб может менять API)
            endpoints = [
                "https://hub.opengradient.ai/api/models",
                "https://hub.opengradient.ai/api/v1/models", 
                "https://api.opengradient.ai/v1/models",
                "https://hub.opengradient.ai/models/list"
            ]
            
            for url in endpoints:
                try:
                    logger.info(f"🔍 Trying endpoint: {url}")
                    resp = await client.get(url)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Парсим разные форматы ответа
                        items = []
                        if isinstance(data, list):
                            items = data
                        elif isinstance(data, dict):
                            items = data.get('data', data.get('models', data.get('items', [])))
                        
                        for item in items:
                            try:
                                # Поддерживаем разные структуры
                                repo = item.get('model_repository', item.get('repository', item.get('model', item)))
                                if isinstance(repo, dict):
                                    model = {
                                        "id": repo.get('name', repo.get('id', f"og-live-{len(live_models)}")),
                                        "name": repo.get('name', repo.get('model_name', 'Unknown Model')),
                                        "description": repo.get('description', repo.get('summary', '')),
                                        "category": repo.get('category', repo.get('type', 'Uncategorized')),
                                        "tags": repo.get('tags', repo.get('labels', [])),
                                        "author": repo.get('author', repo.get('owner', 'OpenGradient')),
                                        "stats": {
                                            "likes": repo.get('stats', {}).get('likes', repo.get('likes', 0)),
                                            "inferences": repo.get('stats', {}).get('inferences', repo.get('downloads', 0))
                                        },
                                        "created_at": repo.get('created_at', repo.get('created', datetime.now().isoformat())),
                                        "is_live": True
                                    }
                                    if model["description"]:  # Только модели с описанием
                                        live_models.append(model)
                            except Exception as e:
                                logger.warning(f"⚠️ Parse error: {e}")
                                continue
                        
                        if live_models:
                            logger.info(f"✅ Fetched {len(live_models)} models from {url}")
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Endpoint {url} failed: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"❌ Fetch error: {e}")
    
    return live_models[:10]  # 🔥 Максимум 10 моделей за раз

async def sync_models_task():
    """🔥 Фоновая задача синхронизации"""
    logger.info("🔄 Starting models sync...")
    
    try:
        sync_status["last_sync"] = datetime.now().isoformat()
        sync_status["errors"] = []
        
        # Получаем живые модели
        live_models = await fetch_live_models_from_hub()
        if not live_models:
            logger.warning("⚠️ No live models fetched")
            sync_status["errors"].append("No models from hub")
            return
        
        logger.info(f"📦 Got {len(live_models)} live models")
        
        if not (DBModel and SessionLocal):
            logger.warning("⚠️ No database - skipping save")
            return
        
        db = SessionLocal()
        added = 0
        
        for lm in live_models:
            try:
                # Проверяем дубликаты
                existing = db.query(DBModel).filter(DBModel.id == lm["id"]).first()
                if existing:
                    continue  # Уже есть
                
                # Сохраняем новую модель
                db_model = DBModel(
                    id=lm["id"],
                    name=lm["name"],
                    description=lm["description"],
                    category=lm["category"],
                    tags=json.dumps(lm["tags"]),
                    stats=json.dumps(lm["stats"]),
                    created_at=lm.get("created_at"),
                    is_live=True,
                    is_user_created=False,
                    synced_at=datetime.now()
                )
                db.add(db_model)
                added += 1
                
                # 🔥 Останавливаемся после 5-10 новых моделей
                if added >= 7:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Save error for {lm.get('id')}: {e}")
                sync_status["errors"].append(f"Save: {lm.get('id')}")
                continue
        
        db.commit()
        db.close()
        
        sync_status["models_added"] = added
        logger.info(f"✅ Sync complete: +{added} new models")
        
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")
        sync_status["errors"].append(f"Sync: {str(e)}")

def start_scheduler():
    """🔥 Запускает планировщик"""
    if not scheduler.running:
        # 🔥 Синхронизация каждые 24 часа
        scheduler.add_job(
            sync_models_task,
            'interval',
            hours=24,
            id='sync_live_models',
            replace_existing=True
        )
        # 🔥 Также запускаем сразу при старте
        scheduler.add_job(
            sync_models_task,
            'date',
            run_date=datetime.now() + timedelta(seconds=10),
            id='sync_on_startup',
            replace_existing=True
        )
        scheduler.start()
        logger.info("📅 Scheduler started - sync every 24h")

# === GEMINI AI CHAT ===

async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            context = f"You are an AI assistant for OpenGradient Model Hub.\n\n"
            if model:
                source = "🔥 LIVE from hub.opengradient.ai" if model.is_live else "📦 Base model"
                context += f"""MODEL INFO ({source}):
Name: {model.name}
ID: {model.id}
Category: {model.category}
Description: {model.description}
Tags: {', '.join(model.tags)}
Stats: {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences

"""
            context += f"USER QUESTION: {query}\n\nAnswer helpfully and specifically about this model."
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": context}]}]}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data['candidates'][0]['content']['parts'][0]['text'].strip()
                logger.error(f"Gemini API error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Gemini exception: {e}")
    
    # Fallback
    if model:
        source = "🔥 LIVE" if model.is_live else "📦"
        return f"{source} **{model.name}**\n\n📋 {model.description}\n🏷️ {model.category}\n🔖 {', '.join(model.tags)}\n📊 {model.stats.get('likes', 0)} likes\n\nAsk me anything!"
    return "👋 Select a model to chat!"

# === СОЗДАНИЕ МОДЕЛЕЙ ===

async def process_model_creation(task_id: str, req: CreateModelRequest):
    import asyncio
    try:
        model_tasks[task_id]["status"] = "processing"
        for progress in [25, 50, 75, 100]:
            model_tasks[task_id]["progress"] = progress
            await asyncio.sleep(1)
        
        model_id = f"og-{req.name.lower().replace(' ', '-')}"
        new_model = ModelInfo(
            id=model_id, name=req.name, description=req.description, category=req.category,
            tags=["custom", "user-created", req.category.lower()],
            stats={"likes": 0, "inferences": 0},
            created_at=datetime.now().isoformat()
        )
        
        if DBModel and SessionLocal:
            db = SessionLocal()
            try:
                db_model = DBModel(
                    id=new_model.id, name=new_model.name, description=new_model.description,
                    category=new_model.category, tags=json.dumps(new_model.tags),
                    stats=json.dumps(new_model.stats), created_at=new_model.created_at,
                    is_live=False, is_user_created=True, synced_at=datetime.now()
                )
                db.add(db_model)
                db.commit()
                logger.info(f"✅ User model saved: {model_id}")
            finally:
                db.close()
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Model created",
            "tx_hash": "0x" + os.urandom(32).hex() if req.wallet_key else None
        }
        
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)

# === ROUTES ===

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "4.0.0-LIVE",
        "timestamp": datetime.now().isoformat(),
        "total_models": len(get_all_models()),
        "base_models": len(BASE_MODELS),
        "database": "✓" if DBModel else "✗",
        "gemini_key": "✓" if os.getenv("GEMINI_API_KEY") else "✗",
        "scheduler": "✓" if scheduler.running else "✗",
        "sync_status": sync_status
    }

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None, limit: int = 100, live_only: bool = False):
    models = get_all_models()
    
    if live_only:
        models = [m for m in models if m.is_live]
    
    if category and category != 'all':
        models = [m for m in models if m.category.lower() == category.lower()]
    
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower() or any(s in t for t in m.tags)]
    
    return [m.model_dump() for m in models[:limit]]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    for m in get_all_models():
        if m.id == model_id:
            return m.model_dump()
    raise HTTPException(404, "Model not found")

@app.get("/api/categories")
async def get_categories():
    models = get_all_models()
    cats = {}
    for m in models:
        cats[m.category] = cats.get(m.category, 0) + 1
    return {"categories": [{"id": k.lower().replace(" ", "-"), "name": k, "count": v} for k, v in cats.items()]}

@app.get("/api/stats")
async def get_stats():
    models = get_all_models()
    live = [m for m in models if m.is_live]
    return {
        "total_models": len(models),
        "live_models": len(live),
        "base_models": len(BASE_MODELS),
        "total_likes": sum(m.stats.get('likes', 0) for m in models),
        "total_inferences": sum(m.stats.get('inferences', 0) for m in models),
        "categories": len(set(m.category for m in models)),
        "last_sync": sync_status.get("last_sync"),
        "sync_added": sync_status.get("models_added", 0)
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model = next((m for m in get_all_models() if m.id == req.model_id), None) if req.model_id else None
    reply = await generate_ai_response(req.query, model)
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    chat_sessions[sid].append({"role": "user", "content": req.query})
    chat_sessions[sid].append({"role": "assistant", "content": reply})
    return {"reply": reply, "session_id": sid}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status": "queued", "progress": 0}
    bg.add_task(process_model_creation, tid, req)
    return {"task_id": tid, "status": "queued"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t

@app.post("/api/sync")
async def trigger_sync():
    """🔥 Ручной запуск синхронизации (для тестов)"""
    await sync_models_task()
    return {"status": "synced", "added": sync_status.get("models_added", 0)}

@app.on_event("startup")
async def startup():
    """🔥 Запуск при старте приложения"""
    logger.info("🚀 OpenGradient Catalog v4.0-LIVE starting...")
    start_scheduler()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
