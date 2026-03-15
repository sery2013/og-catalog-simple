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

app = FastAPI(title="OpenGradient Catalog", version="2.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === МОДЕЛИ ===

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    id: str
    name: str
    description: str
    category: str
    tags: List[str] = []
    author: str = "OpenGradient"
    license: str = "MIT"
    created_at: Optional[str] = None
    stats: Dict[str, int] = {}
    files: List[str] = []
    format: Optional[str] = None

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
    config: Dict[str, Any] = {}
    wallet_key: Optional[str] = None

# === КЭШ МОДЕЛЕЙ ===
models_cache = []
cache_timestamp = None

# === OPENGRADIENT API ===
OG_API_URL = "https://hub.opengradient.ai"

async def fetch_models_from_api():
    """Получаем реальные модели с hub.opengradient.ai"""
    global models_cache, cache_timestamp
    
    # Кэшируем на 1 час
    if cache_timestamp and (datetime.now() - cache_timestamp).seconds < 3600:
        return models_cache
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Пробуем разные endpoints
            endpoints = [
                "/api/models",
                "/models/api/list",
                "/api/v1/models"
            ]
            
            for endpoint in endpoints:
                try:
                    response = await client.get(f"{OG_API_URL}{endpoint}")
                    if response.status_code == 200:
                        data = response.json()
                        models_cache = parse_models_data(data)
                        cache_timestamp = datetime.now()
                        logger.info(f"Fetched {len(models_cache)} models from OpenGradient API")
                        return models_cache
                except Exception as e:
                    logger.warning(f"Endpoint {endpoint} failed: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error fetching from OpenGradient: {e}")
    
    # Fallback на расширенные тестовые данные
    models_cache = get_extended_test_models()
    cache_timestamp = datetime.now()
    return models_cache

def parse_models_data(data):
    """Парсим данные API в нашу структуру"""
    models = []
    
    # Разные форматы ответов
    items = data.get('data', data.get('models', data if isinstance(data, list) else []))
    
    for item in items:
        try:
            if isinstance(item, dict):
                repo = item.get('model_repository', item.get('repository', item))
                releases = item.get('releases', [])
                
                model = ModelInfo(
                    id=repo.get('name', repo.get('id', f"model_{len(models)}")),
                    name=repo.get('name', repo.get('model_name', 'Unknown')),
                    description=repo.get('description', ''),
                    category=repo.get('category', 'Uncategorized'),
                    tags=repo.get('tags', []),
                    author=repo.get('author', 'OpenGradient'),
                    license=repo.get('license', 'MIT'),
                    created_at=repo.get('created_at'),
                    stats={
                        'likes': repo.get('stats', {}).get('likes', 0),
                        'inferences': repo.get('stats', {}).get('inferences', 0)
                    },
                    files=[f.get('filename', f.get('name', '')) for f in releases[0].get('files', [])] if releases else [],
                    format=releases[0].get('format') if releases else None
                )
                models.append(model)
        except Exception as e:
            logger.warning(f"Error parsing model: {e}")
            continue
    
    return models

def get_extended_test_models():
    """Расширенные тестовые данные (fallback)"""
    return [
        ModelInfo(id="og-1hr-volatility-ethusdt", name="ETH/USDT 1hr Volatility", description="Real-time ETH/USDT volatility forecasting model trained on 1-hour OHLCV data using GARCH architecture", category="Risk", tags=["defi", "prediction", "timeseries", "ethereum", "volatility"], stats={"likes": 42, "inferences": 1287}),
        ModelInfo(id="og-llama3-fintune-v2", name="Llama3 Financial Fine-tuned", description="Fine-tuned Llama 3 8B model for financial analysis, sentiment detection, and market prediction", category="Language", tags=["llm", "nlp", "finance", "sentiment", "llama3"], stats={"likes": 156, "inferences": 5432}),
        ModelInfo(id="og-risk-bert-base", name="Risk Assessment BERT", description="BERT-based model for DeFi risk scoring, sybil detection, and protocol safety analysis", category="Risk", tags=["risk", "defi", "classification", "security", "bert"], stats={"likes": 89, "inferences": 3421}),
        ModelInfo(id="og-defi-gemma", name="DeFi Gemma Assistant", description="Gemma 7B model specialized in DeFi protocols, yield farming strategies, and liquidity optimization", category="DeFi", tags=["defi", "llm", "yield", "protocols", "gemma"], stats={"likes": 203, "inferences": 8765}),
        ModelInfo(id="og-amm-optimizer", name="AMM Fee Optimizer", description="Optimizes trading fees for automated market makers based on volatility and volume", category="Protocol", tags=["defi", "amm", "optimization", "trading", "fees"], stats={"likes": 67, "inferences": 2100}),
        ModelInfo(id="og-sybil-detector", name="Sybil Detection Model", description="Graph neural network for detecting sybil attacks in DeFi protocols", category="Risk", tags=["security", "defi", "gnn", "sybil", "detection"], stats={"likes": 134, "inferences": 4567}),
        ModelInfo(id="og-sentiment-crypto", name="Crypto Sentiment Analyzer", description="Real-time sentiment analysis for crypto markets from social media and news", category="Language", tags=["sentiment", "nlp", "crypto", "social", "analysis"], stats={"likes": 178, "inferences": 6543}),
        ModelInfo(id="og-liquidation-predictor", name="Liquidation Predictor", description="Predicts liquidation events in lending protocols 30min in advance", category="Risk", tags=["defi", "lending", "prediction", "risk", "liquidation"], stats={"likes": 95, "inferences": 3210}),
        ModelInfo(id="og-yield-optimizer", name="Yield Farming Optimizer", description="Optimizes yield farming strategies across multiple protocols", category="DeFi", tags=["defi", "yield", "optimization", "farming", "strategy"], stats={"likes": 221, "inferences": 7890}),
        ModelInfo(id="og-nft-pricer", name="NFT Price Predictor", description="Predicts NFT floor prices using on-chain and market data", category="Multimodal", tags=["nft", "prediction", "pricing", "marketplace"], stats={"likes": 112, "inferences": 2890}),
        ModelInfo(id="og-mev-detector", name="MEV Opportunity Detector", description="Detects MEV opportunities in real-time from mempool data", category="Protocol", tags=["mev", "trading", "arbitrage", "defi", "mempool"], stats={"likes": 187, "inferences": 5670}),
        ModelInfo(id="og-portfolio-advisor", name="DeFi Portfolio Advisor", description="AI advisor for DeFi portfolio optimization and risk management", category="Language", tags=["defi", "portfolio", "advisor", "optimization", "risk"], stats={"likes": 156, "inferences": 4320}),
    ]

# === ЧАТ С AI ===
chat_sessions: Dict[str, List] = {}

async def generate_ai_response(query: str, model_info: Optional[ModelInfo] = None) -> str:
    """Генерируем ответ с помощью реального AI (OpenAI/Gemini)"""
    
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        # Fallback на шаблонные ответы
        return generate_template_response(query, model_info)
    
    try:
        # Формируем контекст
        context = f"""You are an AI assistant for OpenGradient Model Hub.
You help users understand AI models for blockchain and DeFi.

"""
        if model_info:
            context += f"""Current model context:
- Name: {model_info.name}
- Category: {model_info.category}
- Description: {model_info.description}
- Tags: {', '.join(model_info.tags)}
- Stats: {model_info.stats.get('likes', 0)} likes, {model_info.stats.get('inferences', 0)} inferences
- Format: {model_info.format or 'ONNX'}
- Files: {', '.join(model_info.files) if model_info.files else 'model.onnx, config.json'}

"""
        context += f"""User question: {query}

Provide a helpful, concise answer about the model or the platform."""

        # OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": context},
                            {"role": "user", "content": query}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                data = response.json()
                return data['choices'][0]['message']['content'].strip()
        
        # Gemini API
        elif os.getenv("GEMINI_API_KEY"):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={os.getenv('GEMINI_API_KEY')}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": context}]}]}
                )
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text'].strip()
                
    except Exception as e:
        logger.error(f"AI API error: {e}")
        return generate_template_response(query, model_info)

def generate_template_response(query: str, model_info: Optional[ModelInfo]) -> str:
    """Шаблонные ответы (fallback)"""
    q = query.lower()
    
    if model_info:
        if any(word in q for word in ["how", "work", "architecture"]):
            return f"The **{model_info.name}** uses advanced machine learning techniques optimized for {model_info.category} tasks.\n\n**Key Features:**\n• Trained on real blockchain data\n• Optimized for ONNX runtime\n• {len(model_info.tags)} specialized tags: {', '.join(model_info.tags)}\n\n**Performance:**\n👍 {model_info.stats.get('likes', 0)} users liked this model\n🔄 {model_info.stats.get('inferences', 0)} total inferences"
        
        elif any(word in q for word in ["use", "deploy", "run"]):
            return f"To use **{model_info.name}**:\n\n1. **Download** the model files from the repository\n2. **Install** ONNX runtime: `pip install onnxruntime`\n3. **Load** the model in your code\n4. **Send** input data in the expected format\n5. **Get** predictions\n\nThe model is ready for production use!"
        
        elif any(word in q for word in ["accuracy", "performance", "metric"]):
            return f"**{model_info.name}** Performance Metrics:\n\n• **User Rating:** {model_info.stats.get('likes', 0)} likes\n• **Usage:** {model_info.stats.get('inferences', 0)} inferences\n• **Category:** {model_info.category}\n• **Format:** {model_info.format or 'ONNX'}\n\nThis model is actively used in production environments."
        
        else:
            return f"About **{model_info.name}**:\n\n📋 **Description:** {model_info.description}\n\n🏷️ **Category:** {model_info.category}\n\n🔖 **Tags:** {', '.join(model_info.tags)}\n\n📊 **Stats:** {model_info.stats.get('likes', 0)} likes, {model_info.stats.get('inferences', 0)} inferences\n\nWhat else would you like to know?"
    else:
        return "👋 Welcome to OpenGradient Catalog!\n\nI can help you:\n• Find the right model for your use case\n• Understand model capabilities\n• Deploy models on-chain\n• Optimize performance\n\nSelect a model from the catalog to start chatting about it!"

# === СОЗДАНИЕ МОДЕЛЕЙ ===
model_tasks: Dict[str, Dict] = {}

async def process_model_creation(task_id: str, request: CreateModelRequest):
    """Фоновая задача создания модели"""
    import asyncio
    
    model_tasks[task_id]["status"] = "processing"
    model_tasks[task_id]["progress"] = 25
    
    await asyncio.sleep(2)
    model_tasks[task_id]["progress"] = 50
    
    # Здесь была бы реальная интеграция с OpenGradient API
    # Для сейчас симулируем
    await asyncio.sleep(2)
    model_tasks[task_id]["progress"] = 75
    
    await asyncio.sleep(1)
    model_tasks[task_id]["status"] = "completed"
    model_tasks[task_id]["progress"] = 100
    model_tasks[task_id]["result"] = {
        "model_id": f"og-{request.name.lower().replace(' ', '-')}",
        "message": "Model successfully created and deployed",
        "tx_hash": "0x" + os.urandom(32).hex() if request.wallet_key else None
    }

# === ROUTES ===

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_cached": len(models_cache),
        "cache_age": (datetime.now() - cache_timestamp).seconds if cache_timestamp else None
    }

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None, limit: int = 50):
    models = await fetch_models_from_api()
    
    if category and category != 'all':
        models = [m for m in models if m.category.lower() == category.lower()]
    
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower() or any(s in t for t in m.tags)]
    
    return [m.model_dump() for m in models[:limit]]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    models = await fetch_models_from_api()
    for m in models:
        if m.id == model_id:
            return m.model_dump()
    raise HTTPException(404, "Model not found")

@app.get("/api/categories")
async def get_categories():
    models = await fetch_models_from_api()
    categories = {}
    for m in models:
        cat = m.category
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "categories": [
            {"id": "defi", "name": "DeFi", "count": categories.get("DeFi", 0)},
            {"id": "language", "name": "Language Models", "count": categories.get("Language", 0)},
            {"id": "risk", "name": "Risk Models", "count": categories.get("Risk", 0)},
            {"id": "multimodal", "name": "Multimodal", "count": categories.get("Multimodal", 0)},
            {"id": "protocol", "name": "Protocol Optimization", "count": categories.get("Protocol", 0)},
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """Статистика в реальном времени"""
    models = await fetch_models_from_api()
    total_likes = sum(m.stats.get('likes', 0) for m in models)
    total_inferences = sum(m.stats.get('inferences', 0) for m in models)
    
    return {
        "total_models": len(models),
        "total_likes": total_likes,
        "total_inferences": total_inferences,
        "categories": len(set(m.category for m in models)),
        "active_users": len(chat_sessions),
        "cache_timestamp": cache_timestamp.isoformat() if cache_timestamp else None
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    
    # Получаем информацию о модели
    model_info = None
    if req.model_id:
        models = await fetch_models_from_api()
        model_info = next((m for m in models if m.id == req.model_id), None)
    
    # Генерируем AI ответ
    reply = await generate_ai_response(req.query, model_info)
    
    # Сохраняем сессию
    if sid not in chat_sessions:
        chat_sessions[sid] = []
    chat_sessions[sid].append({
        "role": "user",
        "content": req.query,
        "timestamp": datetime.now().isoformat()
    })
    chat_sessions[sid].append({
        "role": "assistant",
        "content": reply,
        "timestamp": datetime.now().isoformat()
    })
    
    return {"reply": reply, "session_id": sid}

@app.get("/api/chat/{session_id}")
async def get_chat(session_id: str):
    return {"session_id": session_id, "messages": chat_sessions.get(session_id, [])}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    
    model_tasks[tid] = {
        "status": "queued",
        "progress": 0,
        "created": datetime.now().isoformat(),
        "request": req.model_dump()
    }
    
    bg.add_task(process_model_creation, tid, req)
    
    return {"task_id": tid, "status": "queued", "message": "Model creation started"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t

@app.delete("/api/chat/{session_id}")
async def delete_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting OpenGradient Catalog API v2.0")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
