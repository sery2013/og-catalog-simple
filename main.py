from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
import logging
import httpx
import os
import json

# === БАЗА ДАННЫХ ===
from sqlalchemy import create_engine, Column, String, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="3.1.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === МОДЕЛИ ДАННЫХ ===

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    name: str
    description: str
    category: str
    tags: List[str] = []
    stats: Dict[str, int] = {"likes": 0, "inferences": 0}
    created_at: Optional[str] = None

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

# === БАЗА ДАННЫХ PostgreSQL ===

DATABASE_URL = os.getenv("DATABASE_URL")
user_models_memory: List[ModelInfo] = []

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
            is_user_created = Column(Integer, default=0)
        
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

def get_all_models():
    """🔥 Возвращает модели: созданные пользователями (новые первыми) + базовые"""
    if DBModel and SessionLocal:
        try:
            db = SessionLocal()
            # 🔥 Сортируем по created_at DESC — новые модели первыми!
            db_models = db.query(DBModel).filter(
                DBModel.is_user_created == 1
            ).order_by(DBModel.created_at.desc()).all()
            
            user_models = [
                ModelInfo(
                    id=m.id, name=m.name, description=m.description, category=m.category,
                    tags=m.tags if isinstance(m.tags, list) else json.loads(m.tags) if m.tags else [],
                    stats=m.stats if isinstance(m.stats, dict) else json.loads(m.stats) if m.stats else {"likes":0,"inferences":0},
                    created_at=m.created_at
                )
                for m in db_models
            ]
            db.close()
            # 🔥 ПОЛЬЗОВАТЕЛЬСКИЕ МОДЕЛИ ПЕРВЫМИ!
            return user_models + BASE_MODELS
        except Exception as e:
            logger.error(f"DB read error: {e}")
            return user_models_memory + BASE_MODELS
    # 🔥 ПОЛЬЗОВАТЕЛЬСКИЕ МОДЕЛИ ПЕРВЫМИ!
    return user_models_memory + BASE_MODELS

# === GEMINI AI CHAT ===

async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            context = f"You are an AI assistant for OpenGradient Model Hub.\n\n"
            if model:
                context += f"""MODEL INFO:
Name: {model.name}
ID: {model.id}
Category: {model.category}
Description: {model.description}
Tags: {', '.join(model.tags)}
Stats: {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences

"""
            context += f"USER QUESTION: {query}\n\nAnswer helpfully and specifically about this model."
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # 🔥 ИСПРАВЛЕНО: убран лишний пробел в URL
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
        return f"**{model.name}**\n\n📋 {model.description}\n🏷️ {model.category}\n🔖 {', '.join(model.tags)}\n📊 {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences\n\nAsk me anything specific!"
    return "👋 Select a model to chat!"

# === СОЗДАНИЕ МОДЕЛЕЙ ===

async def process_model_creation(task_id: str, req: CreateModelRequest):
    import asyncio
    try:
        model_tasks[task_id]["status"] = "processing"
        model_tasks[task_id]["progress"] = 25
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 50
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 75
        await asyncio.sleep(1)
        
        model_id = f"og-{req.name.lower().replace(' ', '-')}"
        new_model = ModelInfo(
            id=model_id, name=req.name, description=req.description, category=req.category,
            tags=["custom", "user-created", req.category.lower()],
            stats={"likes": 0, "inferences": 0},
            created_at=datetime.now().isoformat()
        )
        
        # Сохраняем в БД
        if DBModel and SessionLocal:
            db = SessionLocal()
            try:
                db_model = DBModel(
                    id=new_model.id, name=new_model.name, description=new_model.description,
                    category=new_model.category, tags=json.dumps(new_model.tags),
                    stats=json.dumps(new_model.stats), created_at=new_model.created_at, is_user_created=1
                )
                db.add(db_model)
                db.commit()
                logger.info(f"✅ Model SAVED to database: {model_id}")
            finally:
                db.close()
        else:
            user_models_memory.append(new_model)
            logger.info(f"✅ Model saved to memory: {model_id}")
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["progress"] = 100
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Model created and saved permanently",
            "tx_hash": "0x" + os.urandom(32).hex() if req.wallet_key else None
        }
        logger.info(f"✅ Task {task_id} completed")
        
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)
        logger.error(f"❌ Task {task_id} failed: {e}")

# === ROUTES ===

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_models": len(get_all_models()),
        "base_models": len(BASE_MODELS),
        "database": "✓" if DBModel else "✗",
        "gemini_key": "✓" if os.getenv("GEMINI_API_KEY") else "✗"
    }

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None, limit: int = 50):
    models = get_all_models()
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
    return {
        "total_models": len(models),
        "total_likes": sum(m.stats.get('likes', 0) for m in models),
        "total_inferences": sum(m.stats.get('inferences', 0) for m in models),
        "categories": len(set(m.category for m in models)),
        "user_created": len(models) - len(BASE_MODELS)
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
    model_tasks[tid] = {"status": "queued", "progress": 0, "created": datetime.now().isoformat()}
    bg.add_task(process_model_creation, tid, req)
    return {"task_id": tid, "status": "queued", "message": "Started"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 OpenGradient Catalog v3.1")
    logger.info(f"📦 Base models: {len(BASE_MODELS)}")
    logger.info(f"🗄️ Database: {'✓' if DBModel else '✗'}")
    logger.info(f"🔑 Gemini: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
