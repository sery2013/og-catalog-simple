from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import httpx
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="2.2.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# === БАЗА МОДЕЛЕЙ (в памяти) ===
# Стартовые модели
BASE_MODELS = [
    ModelInfo(id="og-1hr-volatility-ethusdt", name="ETH/USDT 1hr Volatility", description="Real-time ETH/USDT volatility forecasting model trained on 1-hour OHLCV data", category="Risk", tags=["defi", "prediction", "timeseries", "ethereum"], stats={"likes": 42, "inferences": 1287}),
    ModelInfo(id="og-llama3-fintune-v2", name="Llama3 Financial Fine-tuned", description="Fine-tuned Llama 3 8B model for financial analysis, sentiment detection, and market prediction", category="Language", tags=["llm", "nlp", "finance", "sentiment", "llama3"], stats={"likes": 156, "inferences": 5432}),
    ModelInfo(id="og-risk-bert-base", name="Risk Assessment BERT", description="BERT-based model for DeFi risk scoring, sybil detection, and protocol safety analysis", category="Risk", tags=["risk", "defi", "classification", "security", "bert"], stats={"likes": 89, "inferences": 3421}),
    ModelInfo(id="og-defi-gemma", name="DeFi Gemma Assistant", description="Gemma 7B model specialized in DeFi protocols, yield farming strategies, and liquidity optimization", category="DeFi", tags=["defi", "llm", "yield", "protocols", "gemma"], stats={"likes": 203, "inferences": 8765}),
    ModelInfo(id="og-amm-optimizer", name="AMM Fee Optimizer", description="Optimizes trading fees for automated market makers based on volatility and volume", category="Protocol", tags=["defi", "amm", "optimization", "trading", "fees"], stats={"likes": 67, "inferences": 2100}),
    ModelInfo(id="og-sybil-detector", name="Sybil Detection Model", description="Graph neural network for detecting sybil attacks in DeFi protocols", category="Risk", tags=["security", "defi", "gnn", "sybil"], stats={"likes": 134, "inferences": 4567}),
    ModelInfo(id="og-sentiment-crypto", name="Crypto Sentiment Analyzer", description="Real-time sentiment analysis for crypto markets from social media and news", category="Language", tags=["sentiment", "nlp", "crypto", "social"], stats={"likes": 178, "inferences": 6543}),
    ModelInfo(id="og-liquidation-predictor", name="Liquidation Predictor", description="Predicts liquidation events in lending protocols 30min in advance", category="Risk", tags=["defi", "lending", "prediction", "risk"], stats={"likes": 95, "inferences": 3210}),
    ModelInfo(id="og-yield-optimizer", name="Yield Farming Optimizer", description="Optimizes yield farming strategies across multiple protocols", category="DeFi", tags=["defi", "yield", "optimization", "farming"], stats={"likes": 221, "inferences": 7890}),
    ModelInfo(id="og-nft-pricer", name="NFT Price Predictor", description="Predicts NFT floor prices using on-chain and market data", category="Multimodal", tags=["nft", "prediction", "pricing", "marketplace"], stats={"likes": 112, "inferences": 2890}),
    ModelInfo(id="og-mev-detector", name="MEV Opportunity Detector", description="Detects MEV opportunities in real-time from mempool data", category="Protocol", tags=["mev", "trading", "arbitrage", "defi"], stats={"likes": 187, "inferences": 5670}),
    ModelInfo(id="og-portfolio-advisor", name="DeFi Portfolio Advisor", description="AI advisor for DeFi portfolio optimization and risk management", category="Language", tags=["defi", "portfolio", "advisor", "optimization"], stats={"likes": 156, "inferences": 4320}),
]

# Создаемые пользователями модели (сохраняются в памяти)
user_models: List[ModelInfo] = []

# === ХРАНИЛИЩА ===
chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}

# === ФУНКЦИИ ===

def get_all_models():
    """Возвращает все модели (базовые + созданные пользователями)"""
    return BASE_MODELS + user_models

async def generate_ai_response(query: str, model_info: Optional[ModelInfo] = None) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            context = f"You are an AI assistant for OpenGradient Model Hub.\n"
            if model_info:
                context += f"Model: {model_info.name} ({model_info.category})\nDescription: {model_info.description}\nTags: {', '.join(model_info.tags)}\n"
            context += f"User question: {query}\n\nAnswer concisely and helpfully."
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": context}]}]}
                )
                if response.status_code == 200:
                    data = response.json()
                    return data['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
    
    return generate_template_response(query, model_info)

def generate_template_response(query: str, model_info: Optional[ModelInfo]) -> str:
    q = query.lower()
    if model_info:
        if any(w in q for w in ["how", "work", "architecture"]):
            return f"**{model_info.name}** uses advanced ML for {model_info.category}.\n\n**Features:**\n• Blockchain-trained\n• ONNX-optimized\n• Tags: {', '.join(model_info.tags)}\n\n**Stats:** 👍 {model_info.stats.get('likes',0)} | 🔄 {model_info.stats.get('inferences',0)}"
        elif any(w in q for w in ["use", "deploy", "run"]):
            return f"To use **{model_info.name}**:\n1. Download files\n2. `pip install onnxruntime`\n3. Load model\n4. Send input\n5. Get predictions\n\nReady for production!"
        else:
            return f"**{model_info.name}**\n\n📋 {model_info.description}\n🏷️ {model_info.category}\n🔖 {', '.join(model_info.tags)}\n📊 {model_info.stats.get('likes',0)} likes\n\nWhat else?"
    return "👋 Select a model to chat about it!"

async def process_model_creation(task_id: str, request: CreateModelRequest):
    import asyncio
    try:
        model_tasks[task_id]["status"] = "processing"
        model_tasks[task_id]["progress"] = 25
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 50
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 75
        await asyncio.sleep(1)
        
        # Создаём модель и добавляем в память
        model_id = f"og-{request.name.lower().replace(' ', '-')}"
        new_model = ModelInfo(
            id=model_id,
            name=request.name,
            description=request.description,
            category=request.category,
            tags=["custom", "user-created", request.category.lower()],
            stats={"likes": 0, "inferences": 0},
            created_at=datetime.now().isoformat()
        )
        user_models.append(new_model)  # ✅ Сохраняем в памяти!
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["progress"] = 100
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Model created and added to catalog",
            "tx_hash": "0x" + os.urandom(32).hex() if request.wallet_key else None
        }
        logger.info(f"✓ Model created: {model_id}")
        
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)
        logger.error(f"✗ Task {task_id} failed: {e}")

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
        "user_models": len(user_models),
        "gemini_key": "✓" if os.getenv("GEMINI_API_KEY") else "✗",
        "openai_key": "✓" if os.getenv("OPENAI_API_KEY") else "✗"
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
        "user_created": len(user_models)
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model_info = next((m for m in get_all_models() if m.id == req.model_id), None) if req.model_id else None
    reply = await generate_ai_response(req.query, model_info)
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    chat_sessions[sid].append({"role": "user", "content": req.query})
    chat_sessions[sid].append({"role": "assistant", "content": reply})
    return {"reply": reply, "session_id": sid}

@app.get("/api/chat/{session_id}")
async def get_chat(session_id: str):
    return {"session_id": session_id, "messages": chat_sessions.get(session_id, [])}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status": "queued", "progress": 0, "created": datetime.now().isoformat()}
    bg.add_task(process_model_creation, tid, req)
    return {"task_id": tid, "status": "queued", "message": "Model creation started"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t

@app.get("/api/user-models")
async def get_user_models():
    """Получить только модели созданные пользователями"""
    return [m.model_dump() for m in user_models]

@app.delete("/api/user-models/{model_id}")
async def delete_user_model(model_id: str):
    """Удалить модель созданную пользователем"""
    global user_models
    for i, m in enumerate(user_models):
        if m.id == model_id:
            user_models.pop(i)
            return {"status": "deleted", "model_id": model_id}
    raise HTTPException(404, "Model not found")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 OpenGradient Catalog v2.2")
    logger.info(f"Base models: {len(BASE_MODELS)}")
    logger.info(f"User models: {len(user_models)}")
    logger.info(f"Gemini: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
