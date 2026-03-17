from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
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
from sqlalchemy.orm import sessionmaker, Session

# === ПЛАНИРОВЩИК ===
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# === ЛОГИРОВАНИЕ ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="4.1.0-LIVE", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === МОДЕЛИ ДАННЫХ (Pydantic) ===
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
    # ✅ wallet_key УДАЛЁН — больше не требуется

# === 12 БАЗОВЫХ МОДЕЛЕЙ (оригинальный список) ===
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

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ БД ===
DATABASE_URL = os.getenv("DATABASE_URL")
engine = None
SessionLocal = None
DBModel = None
Base = None
db_connection_ok = False  # Флаг: подключена ли БД

# === ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ ===
if DATABASE_URL:
    try:
        logger.info(f"🔗 Connecting to database: {DATABASE_URL[:50]}...")
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,      # ✅ Проверяет соединение перед использованием
            pool_recycle=300,        # ✅ Переподключает каждые 5 минут
            echo=False               # Отключить логирование SQL-запросов
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
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
        db_connection_ok = True
        logger.info("✅ Database connected successfully")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.warning("⚠️ Running in MEMORY-ONLY mode (data will be lost on restart)")
        db_connection_ok = False
else:
    logger.warning("⚠️ No DATABASE_URL environment variable - running in MEMORY-ONLY mode")
    db_connection_ok = False

# === ХРАНИЛИЩА В ПАМЯТИ ===
chat_sessions: Dict[str, List[Dict]] = {}
model_tasks: Dict[str, Dict] = {}
memory_models: Dict[str, Dict] = {}  # Для пользовательских моделей, если БД недоступна
sync_status = {"last_sync": None, "models_added": 0, "errors": []}

# === ПЛАНИРОВЩИК ЗАДАЧ ===
scheduler = AsyncIOScheduler()

# === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: получение сессии БД ===
def get_db():
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

# === ФУНКЦИЯ: получение всех моделей ===
def get_all_models():
    """🔥 Возвращает модели в порядке приоритета: Live → User-created → Base"""
    result = []
    
    # 🔹 1. Live-модели из БД
    if db_connection_ok and DBModel and SessionLocal:
        try:
            db = SessionLocal()
            live_models = db.query(DBModel).filter(DBModel.is_live == True).order_by(DBModel.synced_at.desc()).all()
            for m in live_models:
                try:
                    tags = m.tags if isinstance(m.tags, list) else (json.loads(m.tags) if m.tags else [])
                    stats = m.stats if isinstance(m.stats, dict) else (json.loads(m.stats) if m.stats else {"likes": 0, "inferences": 0})
                    result.append(ModelInfo(
                        id=m.id, name=m.name, description=m.description, category=m.category,
                        tags=tags, stats=stats, created_at=m.created_at, is_live=m.is_live
                    ))
                except Exception as e:
                    logger.warning(f"⚠️ Parse error for live model {m.id}: {e}")
            db.close()
        except Exception as e:
            logger.error(f"⚠️ Failed to read live models from DB: {e}")
            db_connection_ok = False
    
    # 🔹 2. Пользовательские модели (БД или память)
    if db_connection_ok and DBModel and SessionLocal:
        try:
            db = SessionLocal()
            user_models = db.query(DBModel).filter(DBModel.is_user_created == True).all()
            for m in user_models:
                try:
                    tags = m.tags if isinstance(m.tags, list) else (json.loads(m.tags) if m.tags else [])
                    stats = m.stats if isinstance(m.stats, dict) else (json.loads(m.stats) if m.stats else {"likes": 0, "inferences": 0})
                    result.append(ModelInfo(
                        id=m.id, name=m.name, description=m.description, category=m.category,
                        tags=tags, stats=stats, created_at=m.created_at, is_live=m.is_live
                    ))
                except Exception as e:
                    logger.warning(f"⚠️ Parse error for user model {m.id}: {e}")
            db.close()
        except Exception as e:
            logger.error(f"⚠️ Failed to read user models from DB: {e}")
            db_connection_ok = False
    
    # 🔹 Fallback: модели из памяти (если БД недоступна)
    if not db_connection_ok:
        for mid, mdata in memory_models.items():
            try:
                result.append(ModelInfo(**mdata))
            except Exception as e:
                logger.warning(f"⚠️ Parse error for memory model {mid}: {e}")
    
    # 🔹 3. Базовые модели (всегда добавляются)
    result += BASE_MODELS
    return result

# === ФУНКЦИЯ: получение моделей с хаба (с авторизацией) ===
async def fetch_live_models_from_hub() -> List[Dict]:
    """🔥 Получает live-модели с официального хаба OpenGradient"""
    live_models = []
    hub_email = os.getenv("OPENGRADIENT_HUB_EMAIL")
    hub_password = os.getenv("OPENGRADIENT_HUB_PASSWORD")
    
    if not hub_email or not hub_password:
        logger.warning("⚠️ No OPENGRADIENT_HUB_EMAIL/PASSWORD - using public fallback")
        return await fetch_models_public_fallback()
    
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "OpenGradient-Catalog/4.1", "Content-Type": "application/json"}
        ) as client:
            # 🔐 Шаг 1: Аутентификация
            auth_token = None
            auth_endpoints = [
                ("https://hub.opengradient.ai/api/auth/login", {"email": hub_email, "password": hub_password}),
                ("https://hub.opengradient.ai/api/v1/auth/login", {"email": hub_email, "password": hub_password}),
            ]
            for auth_url, auth_data in auth_endpoints:
                try:
                    logger.info(f"🔑 Trying auth: {auth_url}")
                    resp = await client.post(auth_url, json=auth_data)
                    if resp.status_code == 200:
                        data = resp.json()
                        auth_token = data.get('token') or data.get('access_token') or data.get('auth_token')
                        if auth_token:
                            client.headers["Authorization"] = f"Bearer {auth_token}"
                            logger.info("✅ Authenticated successfully")
                            break
                    else:
                        logger.warning(f"⚠️ Auth failed {resp.status_code}: {auth_url}")
                except Exception as e:
                    logger.warning(f"⚠️ Auth exception {auth_url}: {e}")
                    continue
            
            # 📦 Шаг 2: Получение моделей
            endpoints = [
                "https://hub.opengradient.ai/api/models",
                "https://hub.opengradient.ai/api/v1/models",
                "https://hub.opengradient.ai/api/v2/models",
                "https://api.opengradient.ai/v1/models",
            ]
            for url in endpoints:
                try:
                    logger.info(f"🔍 Fetching: {url}")
                    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
                    resp = await client.get(url, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else data.get('data', data.get('models', data.get('items', data.get('result', []))))
                        for item in items:
                            try:
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
                                    if model["description"] and len(model["description"]) > 10:
                                        live_models.append(model)
                            except Exception as e:
                                logger.warning(f"⚠️ Parse error for item: {e}")
                                continue
                        if live_models:
                            logger.info(f"✅✅✅ Fetched {len(live_models)} models from {url}")
                            break
                    else:
                        logger.warning(f"⚠️ Fetch failed {resp.status_code}: {url}")
                except Exception as e:
                    logger.warning(f"⚠️ Fetch exception {url}: {e}")
                    continue
    except Exception as e:
        logger.error(f"❌ Hub fetch error: {e}")
    
    return live_models[:10]  # Ограничиваем до 10 моделей

# === ФУНКЦИЯ: публичный fallback без авторизации ===
async def fetch_models_public_fallback() -> List[Dict]:
    """🔄 Получает модели через публичные эндпоинты (без авторизации)"""
    live_models = []
    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "OpenGradient-Catalog/4.1"}
        ) as client:
            endpoints = [
                "https://hub.opengradient.ai/api/models",
                "https://hub.opengradient.ai/api/v1/models",
                "https://api.opengradient.ai/v1/models",
            ]
            for url in endpoints:
                try:
                    logger.info(f"🔍 Public fallback: {url}")
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else data.get('data', data.get('models', []))
                        for item in items:
                            repo = item.get('model_repository', item.get('repository', item))
                            if isinstance(repo, dict):
                                model = {
                                    "id": repo.get('name', f"og-live-{len(live_models)}"),
                                    "name": repo.get('name', 'Unknown'),
                                    "description": repo.get('description', ''),
                                    "category": repo.get('category', 'Uncategorized'),
                                    "tags": repo.get('tags', []),
                                    "author": repo.get('author', 'OpenGradient'),
                                    "stats": {"likes": 0, "inferences": 0},
                                    "created_at": repo.get('created_at', datetime.now().isoformat()),
                                    "is_live": True
                                }
                                if model["description"] and len(model["description"]) > 10:
                                    live_models.append(model)
                        if live_models:
                            logger.info(f"✅ Fetched {len(live_models)} via public fallback")
                            break
                except Exception as e:
                    logger.warning(f"⚠️ Public fallback failed {url}: {e}")
                    continue
    except Exception as e:
        logger.error(f"❌ Public fallback error: {e}")
    return live_models[:10]

# === ФУНКЦИЯ: фоновая синхронизация моделей ===
async def sync_models_task():
    """🔥 Фоновая задача: синхронизация live-моделей из хаба"""
    logger.info("🔄 Starting models sync task...")
    try:
        sync_status["last_sync"] = datetime.now().isoformat()
        sync_status["models_added"] = 0
        sync_status["errors"] = []
        
        # Получаем live-модели
        live_models = await fetch_live_models_from_hub()
        if not live_models:
            logger.warning("⚠️ No live models fetched from hub")
            sync_status["errors"].append("No models from hub")
            return
        
        logger.info(f"📦 Got {len(live_models)} live models to process")
        
        # Если нет БД — пропускаем сохранение
        if not (db_connection_ok and DBModel and SessionLocal):
            logger.warning("⚠️ Database not available - skipping save")
            return
        
        db = SessionLocal()
        added = 0
        for lm in live_models:
            try:
                # Проверяем дубликаты
                existing = db.query(DBModel).filter(DBModel.id == lm["id"]).first()
                if existing:
                    logger.info(f"⏭️ Model {lm['id']} already exists, skipping")
                    continue
                
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
                logger.info(f"✅ Added live model {added}: {lm['id']}")
                
                # Останавливаемся после 7 моделей (лимит)
                if added >= 7:
                    logger.info(f"🎯 Reached limit of 7 live models, stopping")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Save error for {lm.get('id')}: {e}")
                sync_status["errors"].append(f"Save: {lm.get('id')} - {str(e)}")
                continue
        
        db.commit()
        db.close()
        sync_status["models_added"] = added
        logger.info(f"✅✅✅ SYNC COMPLETE: +{added} new live models")
        
    except Exception as e:
        logger.error(f"❌ Sync task failed: {e}")
        sync_status["errors"].append(f"Sync task: {str(e)}")

# === ФУНКЦИЯ: запуск планировщика ===
def start_scheduler():
    """🔥 Инициализирует и запускает APScheduler"""
    if not scheduler.running:
        # Синхронизация каждые 24 часа
        scheduler.add_job(
            sync_models_task,
            'interval',
            hours=24,
            id='sync_live_models',
            replace_existing=True,
            max_instances=1
        )
        # Запуск сразу при старте (через 10 секунд)
        scheduler.add_job(
            sync_models_task,
            'date',
            run_date=datetime.now() + timedelta(seconds=10),
            id='sync_on_startup',
            replace_existing=True
        )
        scheduler.start()
        logger.info("📅 Scheduler started - sync every 24h + startup")

# === ФУНКЦИЯ: генерация ответа через Gemini ===
async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    """🤖 Генерирует ответ через Google Gemini API"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    # Проверяем валидность ключа
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            # Формируем контекст
            context = f"You are an AI assistant for OpenGradient Model Hub.\n"
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
            context += f"\nUSER QUESTION: {query}\n\nAnswer helpfully and specifically about this model."
            
            # Запрос к Gemini API
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": context}]}]}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data['candidates'][0]['content']['parts'][0]['text'].strip()
                else:
                    logger.error(f"❌ Gemini API error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"❌ Gemini exception: {e}")
    
    # Fallback-ответ, если Gemini недоступен
    if model:
        source = "🔥 LIVE" if model.is_live else "📦"
        return f"{source} **{model.name}**\n📋 {model.description}\n🏷️ {model.category}\n🔖 {', '.join(model.tags)}\n📊 {model.stats.get('likes', 0)} likes\n\nAsk me anything about this model!"
    return "👋 Hello! Select a model from the catalog to start chatting."

# === ФУНКЦИЯ: обработка создания модели (фоновая задача) ===
async def process_model_creation(task_id: str, req: CreateModelRequest):
    """🔧 Фоновая задача: создание пользовательской модели"""
    try:
        model_tasks[task_id]["status"] = "processing"
        model_tasks[task_id]["progress"] = 0
        
        # Имитация прогресса
        for progress in [25, 50, 75, 100]:
            model_tasks[task_id]["progress"] = progress
            await asyncio.sleep(0.5)
        
        # Генерируем уникальный ID модели
        model_id = f"og-{req.name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
        
        # Создаём объект модели
        new_model_data = {
            "id": model_id,
            "name": req.name,
            "description": req.description,
            "category": req.category,
            "tags": ["custom", "user-created", req.category.lower()],
            "stats": {"likes": 0, "inferences": 0},
            "created_at": datetime.now().isoformat(),
            "is_live": False
        }
        
        # Попытка сохранить в БД
        saved_to_db = False
        if db_connection_ok and DBModel and SessionLocal:
            try:
                db = SessionLocal()
                db_model = DBModel(
                    id=new_model_data["id"],
                    name=new_model_data["name"],
                    description=new_model_data["description"],
                    category=new_model_data["category"],
                    tags=json.dumps(new_model_data["tags"]),
                    stats=json.dumps(new_model_data["stats"]),
                    created_at=new_model_data["created_at"],
                    is_live=new_model_data["is_live"],
                    is_user_created=True,
                    synced_at=datetime.now()
                )
                db.add(db_model)
                db.commit()
                db.refresh(db_model)  # ✅ Обновляем объект после коммита
                db.close()
                saved_to_db = True
                logger.info(f"✅ User model saved to DB: {model_id}")
            except Exception as e:
                logger.error(f"❌ DB save error: {e}")
                db_connection_ok = False  # Помечаем БД как недоступную
        
        # Fallback: сохраняем в память
        if not saved_to_db:
            memory_models[model_id] = new_model_data
            logger.info(f"💾 User model saved to memory: {model_id}")
        
        # Завершаем задачу
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Model created successfully",
            "tx_hash": "0x" + os.urandom(32).hex(),  # Всегда генерируем хэш
            "saved_to_db": saved_to_db
        }
        
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)
        logger.error(f"❌ Task {task_id} failed: {e}")

# === ROUTES ===

@app.get("/")
async def root():
    """🏠 Главная страница — отдаёт index.html"""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """🏥 Health check endpoint"""
    models = get_all_models()
    return {
        "status": "healthy",
        "version": "4.1.0-LIVE",
        "timestamp": datetime.now().isoformat(),
        "total_models": len(models),
        "base_models": len(BASE_MODELS),
        "user_models_memory": len(memory_models),
        "database": "✓ connected" if db_connection_ok else "✗ disconnected",
        "gemini_key": "✓" if os.getenv("GEMINI_API_KEY") else "✗",
        "scheduler": "✓ running" if scheduler.running else "✗ stopped",
        "sync_status": sync_status
    }

@app.get("/api/models")
async def list_models(
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100,
    live_only: bool = False
):
    """📋 Список моделей с фильтрацией"""
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
    """🔍 Получение модели по ID"""
    for m in get_all_models():
        if m.id == model_id:
            return m.model_dump()
    raise HTTPException(status_code=404, detail="Model not found")

@app.get("/api/categories")
async def get_categories():
    """🏷️ Список категорий с количеством моделей"""
    models = get_all_models()
    cats = {}
    for m in models:
        cats[m.category] = cats.get(m.category, 0) + 1
    return {
        "categories": [
            {"id": k.lower().replace(" ", "-"), "name": k, "count": v}
            for k, v in sorted(cats.items(), key=lambda x: x[1], reverse=True)
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """📊 Статистика по каталогу"""
    models = get_all_models()
    live = [m for m in models if m.is_live]
    return {
        "total_models": len(models),
        "live_models": len(live),
        "base_models": len(BASE_MODELS),
        "user_models": len([m for m in models if not m.is_live and m.id not in [bm.id for bm in BASE_MODELS]]),
        "total_likes": sum(m.stats.get('likes', 0) for m in models),
        "total_inferences": sum(m.stats.get('inferences', 0) for m in models),
        "categories": len(set(m.category for m in models)),
        "last_sync": sync_status.get("last_sync"),
        "sync_added": sync_status.get("models_added", 0),
        "sync_errors": sync_status.get("errors", [])
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """💬 Chat endpoint с Gemini"""
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model = None
    if req.model_id:
        model = next((m for m in get_all_models() if m.id == req.model_id), None)
    
    reply = await generate_ai_response(req.query, model)
    
    # Сохраняем историю сессии
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    chat_sessions[sid].append({"role": "user", "content": req.query})
    chat_sessions[sid].append({"role": "assistant", "content": reply})
    
    return {"reply": reply, "session_id": sid}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    """🛠 Создание новой модели (асинхронная задача)"""
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status": "queued", "progress": 0, "created_at": datetime.now().isoformat()}
    bg.add_task(process_model_creation, tid, req)
    return {"task_id": tid, "status": "queued", "message": "Model creation started"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """📋 Получение статуса задачи"""
    task = model_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/api/sync")
async def trigger_sync():
    """🔄 Ручной запуск синхронизации (для тестов)"""
    await sync_models_task()
    return {
        "status": "synced",
        "added": sync_status.get("models_added", 0),
        "errors": sync_status.get("errors", []),
        "last_sync": sync_status.get("last_sync")
    }

# === СОБЫТИЯ ПРИЛОЖЕНИЯ ===

@app.on_event("startup")
async def startup_event():
    """🚀 Выполняется при старте приложения"""
    logger.info("🚀 OpenGradient Catalog v4.1-LIVE starting...")
    logger.info(f"🔑 Hub Email: {'✓' if os.getenv('OPENGRADIENT_HUB_EMAIL') else '✗'}")
    logger.info(f"🔑 Gemini Key: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    logger.info(f"🗄️ Database: {'✓' if db_connection_ok else '✗'}")
    start_scheduler()

@app.on_event("shutdown")
async def shutdown_event():
    """🛑 Выполняется при остановке приложения"""
    logger.info("🛑 Shutting down...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("📅 Scheduler stopped")
    if engine:
        engine.dispose()
        logger.info("🗄️ Database connections closed")

# === ЗАПУСК ===

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
