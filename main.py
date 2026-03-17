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

# === БАЗА ДАННЫХ ===
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# === ПЛАНИРОВЩИК ===
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="4.1.0-LIVE", docs_url="/docs")
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
    # ✅ wallet_key УДАЛЁН

# === 12 БАЗОВЫХ МОДЕЛЕЙ ===
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
engine = None
SessionLocal = None
DBModel = None
Base = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
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
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
else:
    logger.warning("⚠️ No DATABASE_URL - using memory only")

# === ХРАНИЛИЩА ===
chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}
sync_status = {"last_sync": None, "models_added": 0, "errors": []}

# === ПЛАНИРОВЩИК ===
scheduler = AsyncIOScheduler()

# === ФУНКЦИИ ===
def get_all_models():
    """🔥 Возвращает: Live-модели → Пользовательские → Базовые"""
    if DBModel and SessionLocal and engine:
        try:
            db = SessionLocal()
            live_models = db.query(DBModel).filter(DBModel.is_live == True).order_by(DBModel.synced_at.desc()).all()
            user_models = db.query(DBModel).filter(DBModel.is_user_created == True).all()
            
            logger.info(f"📊 DB: {len(live_models)} live, {len(user_models)} user-created")
            
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
            result += BASE_MODELS
            db.close()
            return result
        except Exception as e:
            logger.error(f"DB read error: {e}")
            return BASE_MODELS
    return BASE_MODELS

async def fetch_live_models_from_hub() -> List[Dict]:
    """🔥 Получает модели с официального хаба"""
    live_models = []
    hub_email = os.getenv("OPENGRADIENT_HUB_EMAIL")
    hub_password = os.getenv("OPENGRADIENT_HUB_PASSWORD")
    
    if not hub_email or not hub_password:
        logger.warning("⚠️ No hub credentials - using fallback")
        return await fetch_models_public_fallback()
    
    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OpenGradient-Catalog/4.1", "Content-Type": "application/json"}) as client:
            # Аутентификация
            auth_token = None
            auth_endpoints = [
                ("https://hub.opengradient.ai/api/auth/login", {"email": hub_email, "password": hub_password}),
                ("https://hub.opengradient.ai/api/v1/auth/login", {"email": hub_email, "password": hub_password}),
            ]
            for auth_url, auth_data in auth_endpoints:
                try:
                    resp = await client.post(auth_url, json=auth_data)
                    if resp.status_code == 200:
                        data = resp.json()
                        auth_token = data.get('token') or data.get('access_token') or data.get('auth_token')
                        if auth_token:
                            client.headers["Authorization"] = f"Bearer {auth_token}"
                            logger.info("✅ Authenticated")
                            break
                except: continue
            
            # Получение моделей
            endpoints = ["https://hub.opengradient.ai/api/models", "https://hub.opengradient.ai/api/v1/models", "https://api.opengradient.ai/v1/models"]
            for url in endpoints:
                try:
                    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else data.get('data', data.get('models', data.get('items', [])))
                        for item in items:
                            repo = item.get('model_repository', item.get('repository', item.get('model', item)))
                            if isinstance(repo, dict):
                                model = {
                                    "id": repo.get('name', f"og-live-{len(live_models)}"),
                                    "name": repo.get('name', 'Unknown'),
                                    "description": repo.get('description', ''),
                                    "category": repo.get('category', 'Uncategorized'),
                                    "tags": repo.get('tags', []),
                                    "author": repo.get('author', 'OpenGradient'),
                                    "stats": {"likes": repo.get('stats', {}).get('likes', 0), "inferences": repo.get('stats', {}).get('inferences', 0)},
                                    "created_at": repo.get('created_at', datetime.now().isoformat()),
                                    "is_live": True
                                }
                                if model["description"]: live_models.append(model)
                        if live_models: break
                except: continue
    except Exception as e:
        logger.error(f"❌ Fetch error: {e}")
    return live_models[:10]

async def fetch_models_public_fallback() -> List[Dict]:
    """🔄 Fallback без авторизации"""
    live_models = []
    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "OpenGradient-Catalog/4.1"}) as client:
            for url in ["https://hub.opengradient.ai/api/models", "https://api.opengradient.ai/v1/models"]:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else data.get('data', data.get('models', []))
                        for item in items:
                            repo = item.get('model_repository', item.get('repository', item))
                            if isinstance(repo, dict):
                                model = {"id": repo.get('name', f"og-{len(live_models)}"), "name": repo.get('name', 'Unknown'), "description": repo.get('description', ''), "category": repo.get('category', 'Uncategorized'), "tags": repo.get('tags', []), "author": repo.get('author', 'OpenGradient'), "stats": {"likes": 0, "inferences": 0}, "created_at": datetime.now().isoformat(), "is_live": True}
                                if model["description"]: live_models.append(model)
                        if live_models: break
                except: continue
    except: pass
    return live_models[:10]

async def sync_models_task():
    """🔥 Фоновая синхронизация"""
    logger.info("🔄 Starting sync...")
    try:
        sync_status["last_sync"] = datetime.now().isoformat()
        sync_status["models_added"] = 0
        live_models = await fetch_live_models_from_hub()
        if not live_models or not (DBModel and SessionLocal and engine): return
        
        db = SessionLocal()
        added = 0
        for lm in live_models:
            try:
                if db.query(DBModel).filter(DBModel.id == lm["id"]).first(): continue
                db_model = DBModel(id=lm["id"], name=lm["name"], description=lm["description"], category=lm["category"], tags=json.dumps(lm["tags"]), stats=json.dumps(lm["stats"]), created_at=lm.get("created_at"), is_live=True, is_user_created=False, synced_at=datetime.now())
                db.add(db_model)
                added += 1
                if added >= 7: break
            except: continue
        db.commit()
        db.close()
        sync_status["models_added"] = added
        logger.info(f"✅ Sync complete: +{added}")
    except Exception as e:
        logger.error(f"❌ Sync failed: {e}")

def start_scheduler():
    """🔥 Запуск планировщика"""
    if not scheduler.running:
        scheduler.add_job(sync_models_task, 'interval', hours=24, id='sync_live', replace_existing=True)
        scheduler.add_job(sync_models_task, 'date', run_date=datetime.now() + timedelta(seconds=10), id='sync_startup', replace_existing=True)
        scheduler.start()
        logger.info("📅 Scheduler started")

# === GEMINI CHAT ===
async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            context = f"You are an AI assistant for OpenGradient.\n"
            if model:
                context += f"MODEL: {model.name} ({model.category})\n{model.description}\nTags: {', '.join(model.tags)}\n"
            context += f"QUESTION: {query}\nAnswer helpfully."
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}", headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": context}]}]})
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        except: pass
    if model:
        return f"🔥 **{model.name}**\n📋 {model.description}\n🏷️ {model.category}\n📊 {model.stats.get('likes', 0)} likes"
    return "👋 Select a model to chat!"

# === СОЗДАНИЕ МОДЕЛИ ===
async def process_model_creation(task_id: str, req: CreateModelRequest):
    try:
        model_tasks[task_id]["status"] = "processing"
        for p in [25, 50, 75, 100]:
            model_tasks[task_id]["progress"] = p
            await asyncio.sleep(0.5)
        
        model_id = f"og-{req.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
        new_model = ModelInfo(id=model_id, name=req.name, description=req.description, category=req.category, tags=["custom", "user-created"], stats={"likes": 0, "inferences": 0}, created_at=datetime.now().isoformat())
        
        if DBModel and SessionLocal and engine:
            db = SessionLocal()
            try:
                db_model = DBModel(id=new_model.id, name=new_model.name, description=new_model.description, category=new_model.category, tags=json.dumps(new_model.tags), stats=json.dumps(new_model.stats), created_at=new_model.created_at, is_live=False, is_user_created=True, synced_at=datetime.now())
                db.add(db_model)
                db.commit()
                db.refresh(db_model)  # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ
                logger.info(f"✅ Model saved: {model_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"❌ DB error: {e}")
                raise
            finally:
                db.close()
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["result"] = {"model_id": model_id, "message": "Created", "tx_hash": "0x" + os.urandom(32).hex()}
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)
        logger.error(f"❌ Task failed: {e}")

# === ROUTES ===
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "4.1.0-LIVE", "models": len(get_all_models()), "db": "✓" if DBModel else "✗"}

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None, live_only: bool = False):
    models = get_all_models()
    if live_only: models = [m for m in models if m.is_live]
    if category and category != 'all': models = [m for m in models if m.category.lower() == category.lower()]
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower() or any(s in t for t in m.tags)]
    return [m.model_dump() for m in models[:100]]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    for m in get_all_models():
        if m.id == model_id: return m.model_dump()
    raise HTTPException(404, "Not found")

@app.get("/api/stats")
async def get_stats():
    models = get_all_models()
    return {"total_models": len(models), "live_models": len([m for m in models if m.is_live]), "total_likes": sum(m.stats.get('likes', 0) for m in models), "last_sync": sync_status.get("last_sync")}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model = next((m for m in get_all_models() if m.id == req.model_id), None) if req.model_id else None
    reply = await generate_ai_response(req.query, model)
    if sid not in chat_sessions: chat_sessions[sid] = []
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
    if not t: raise HTTPException(404, "Not found")
    return t

@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting OpenGradient Catalog v4.1-LIVE")
    start_scheduler()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
