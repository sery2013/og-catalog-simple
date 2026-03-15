from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
import logging
import httpx
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenGradient Catalog", version="2.3.0", docs_url="/docs")
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

# === ХРАНИЛИЩА ===

user_models: List[ModelInfo] = []  # Сохраняются в памяти
chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}

def get_all_models():
    """Возвращает все модели (12 базовых + созданные пользователями)"""
    return BASE_MODELS + user_models

# === GEMINI AI CHAT ===

async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    """Генерирует ответ через Gemini API"""
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    logger.info(f"🔑 Gemini Key: {'✓ FOUND' if gemini_key else '✗ NOT FOUND'}")
    
    # Пробуем Gemini API
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            # Формируем контекст с информацией о модели
            context = f"""You are an AI assistant for OpenGradient Model Hub.
You help users understand AI models for blockchain and DeFi.

"""
            if model:
                context += f"""CURRENT MODEL INFORMATION:
• Name: {model.name}
• ID: {model.id}
• Category: {model.category}
• Description: {model.description}
• Tags: {', '.join(model.tags)}
• Stats: {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences

"""
            context += f"""USER QUESTION: {query}

Please provide a helpful, specific answer about this model. If the user asks about technical details, explain them clearly. If they ask about use cases, provide practical examples."""

            logger.info(f"📤 Sending to Gemini: {query[:50]}...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": context}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": 512
                        }
                    }
                )
                
                logger.info(f"📥 Gemini Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data['candidates'][0]['content']['parts'][0]['text'].strip()
                    logger.info(f"✅ Gemini Response: {answer[:100]}...")
                    return answer
                else:
                    logger.error(f"❌ Gemini API Error: {response.status_code} - {response.text[:200]}")
                    
        except Exception as e:
            logger.error(f"❌ Gemini Exception: {type(e).__name__}: {e}")
    
    # Fallback на шаблонные ответы (если API не работает)
    logger.info("⚠️ Using fallback template response")
    return generate_template_response(query, model)

def generate_template_response(query: str, model: Optional[ModelInfo]) -> str:
    """Умные шаблонные ответы"""
    q = query.lower()
    
    if model:
        if any(w in q for w in ["how", "work", "architecture", "technical", "algorithm"]):
            return f"**{model.name}** uses advanced machine learning techniques optimized for {model.category} tasks.\n\n**Key Features:**\n• Trained on real blockchain and DeFi data\n• Optimized for ONNX runtime inference\n• Specialized tags: {', '.join(model.tags)}\n• Production-ready with high accuracy\n\n**Performance:**\n👍 {model.stats.get('likes', 0)} users liked this model\n🔄 {model.stats.get('inferences', 0)} total inferences executed"
        
        elif any(w in q for w in ["use", "deploy", "run", "install", "setup", "start"]):
            return f"To deploy **{model.name}**:\n\n1. **Download** the model files from the repository\n2. **Install** ONNX runtime: `pip install onnxruntime`\n3. **Load** the model in your Python code\n4. **Prepare** input data in the expected format\n5. **Run** inference and get predictions\n6. **Integrate** into your application\n\nThe model is optimized for production use with low latency!"
        
        elif any(w in q for w in ["accuracy", "performance", "metric", "score", "precision", "recall"]):
            return f"**{model.name}** Performance Metrics:\n\n• **User Rating:** {model.stats.get('likes', 0)} likes from the community\n• **Usage:** {model.stats.get('inferences', 0)} successful inferences\n• **Category:** {model.category}\n• **Format:** ONNX (optimized)\n• **Tags:** {', '.join(model.tags)}\n\nThis model is actively used in production environments with proven reliability."
        
        elif any(w in q for w in ["what", "describe", "about", "info", "information"]):
            return f"**{model.name}**\n\n📋 **Description:** {model.description}\n\n🏷️ **Category:** {model.category}\n\n🔖 **Tags:** {', '.join(model.tags)}\n\n📊 **Community Stats:**\n• 👍 {model.stats.get('likes', 0)} likes\n• 🔄 {model.stats.get('inferences', 0)} inferences\n\nThis model is part of the OpenGradient decentralized AI ecosystem."
        
        else:
            return f"**About {model.name}:**\n\n{model.description}\n\n**Category:** {model.category}\n**Tags:** {', '.join(model.tags)}\n**Stats:** {model.stats.get('likes', 0)} likes, {model.stats.get('inferences', 0)} inferences\n\nWhat specific aspect would you like to know more about?"
    else:
        return "👋 Welcome to OpenGradient Catalog!\n\nI can help you:\n• Understand model capabilities\n• Find the right model for your use case\n• Learn about deployment options\n• Compare different models\n\nSelect a model from the catalog to start chatting about it!"

# === СОЗДАНИЕ МОДЕЛЕЙ ===

async def process_model_creation(task_id: str, req: CreateModelRequest):
    """Фоновая задача создания модели"""
    import asyncio
    
    try:
        logger.info(f" Starting task {task_id}")
        
        model_tasks[task_id]["status"] = "processing"
        model_tasks[task_id]["progress"] = 25
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 50
        await asyncio.sleep(1)
        
        model_tasks[task_id]["progress"] = 75
        await asyncio.sleep(1)
        
        # Создаём новую модель
        model_id = f"og-{req.name.lower().replace(' ', '-')}"
        new_model = ModelInfo(
            id=model_id,
            name=req.name,
            description=req.description,
            category=req.category,
            tags=["custom", "user-created", req.category.lower()],
            stats={"likes": 0, "inferences": 0},
            created_at=datetime.now().isoformat()
        )
        
        # ✅ СОХРАНЯЕМ В ПАМЯТЬ
        user_models.append(new_model)
        logger.info(f"✅ Model saved to memory: {model_id} (total user models: {len(user_models)})")
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["progress"] = 100
        model_tasks[task_id]["result"] = {
            "model_id": model_id,
            "message": "Model created and added to catalog",
            "tx_hash": "0x" + os.urandom(32).hex() if req.wallet_key else None
        }
        
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
    logger.info(f"💬 Chat request: model_id={req.model_id}, query={req.query[:50]}...")
    
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    
    # Ищем модель
    model = None
    if req.model_id:
        model = next((m for m in get_all_models() if m.id == req.model_id), None)
        logger.info(f"📦 Model found: {model.name if model else 'None'}")
    
    # Генерируем ответ
    reply = await generate_ai_response(req.query, model)
    
    # Сохраняем сессию
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
    logger.info(f"🛠 Create model request: {req.name}")
    
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {
        "status": "queued",
        "progress": 0,
        "created": datetime.now().isoformat()
    }
    
    bg.add_task(process_model_creation, tid, req)
    
    return {"task_id": tid, "status": "queued", "message": "Model creation started"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t

@app.get("/api/user-models")
async def get_user_models_endpoint():
    """Получить только модели созданные пользователями"""
    return [m.model_dump() for m in user_models]

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting OpenGradient Catalog v2.3")
    logger.info(f"📦 Base models: {len(BASE_MODELS)}")
    logger.info(f"👤 User models: {len(user_models)}")
    logger.info(f"🔑 Gemini API: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
