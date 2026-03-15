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

app = FastAPI(title="OpenGradient Catalog", version="2.2.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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

BASE_MODELS = [
    ModelInfo(id="og-1hr-volatility-ethusdt", name="ETH/USDT 1hr Volatility", description="Real-time ETH/USDT volatility forecasting model", category="Risk", tags=["defi", "prediction", "ethereum"], stats={"likes": 42, "inferences": 1287}),
    ModelInfo(id="og-llama3-fintune-v2", name="Llama3 Financial Fine-tuned", description="Fine-tuned Llama 3 8B for financial analysis", category="Language", tags=["llm", "nlp", "finance", "llama3"], stats={"likes": 156, "inferences": 5432}),
    ModelInfo(id="og-risk-bert-base", name="Risk Assessment BERT", description="BERT model for DeFi risk scoring", category="Risk", tags=["risk", "defi", "bert"], stats={"likes": 89, "inferences": 3421}),
    ModelInfo(id="og-defi-gemma", name="DeFi Gemma Assistant", description="Gemma 7B for DeFi protocols", category="DeFi", tags=["defi", "llm", "yield"], stats={"likes": 203, "inferences": 8765}),
    ModelInfo(id="og-amm-optimizer", name="AMM Fee Optimizer", description="Optimizes AMM trading fees", category="Protocol", tags=["defi", "amm", "trading"], stats={"likes": 67, "inferences": 2100}),
]

user_models: List[ModelInfo] = []
chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}

def get_all_models():
    return BASE_MODELS + user_models

async def generate_ai_response(query: str, model: Optional[ModelInfo] = None) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    logger.info(f"Gemini key: {'✓' if gemini_key else '✗'}")
    
    if gemini_key and gemini_key.startswith("AIza"):
        try:
            context = f"You are an AI assistant for OpenGradient. "
            if model:
                context += f"Model: {model.name} ({model.category}). {model.description}. Tags: {', '.join(model.tags)}. "
            context += f"User asks: {query}. Answer helpfully and specifically about this model."
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": context}]}]}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data['candidates'][0]['content']['parts'][0]['text'].strip()
                    logger.info(f"✓ Gemini: {answer[:80]}...")
                    return answer
                else:
                    logger.error(f"✗ Gemini API: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"✗ Gemini exception: {e}")
    
    # Fallback
    logger.info("Using fallback response")
    if model:
        return f"**{model.name}**\n\n📋 {model.description}\n🏷️ {model.category}\n🔖 {', '.join(model.tags)}\n📊 {model.stats['likes']} likes, {model.stats['inferences']} inferences\n\nAsk me anything specific!"
    return "👋 Select a model to chat!"

async def process_creation(task_id: str, req: CreateModelRequest):
    import asyncio
    try:
        model_tasks[task_id]["status"] = "processing"
        for progress in [25, 50, 75, 100]:
            model_tasks[task_id]["progress"] = progress
            await asyncio.sleep(1)
        
        model_id = f"og-{req.name.lower().replace(' ','-')}"
        new_model = ModelInfo(id=model_id, name=req.name, description=req.description, category=req.category, tags=["custom", req.category.lower()], stats={"likes":0,"inferences":0}, created_at=datetime.now().isoformat())
        user_models.append(new_model)
        
        model_tasks[task_id]["status"] = "completed"
        model_tasks[task_id]["result"] = {"model_id": model_id, "message": "Created", "tx_hash": "0x"+os.urandom(32).hex() if req.wallet_key else None}
        logger.info(f"✓ Model created: {model_id}")
    except Exception as e:
        model_tasks[task_id]["status"] = "failed"
        model_tasks[task_id]["error"] = str(e)

@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status":"healthy", "timestamp":datetime.now().isoformat(), "total_models":len(get_all_models()), "base":len(BASE_MODELS), "user":len(user_models), "gemini":"✓" if os.getenv("GEMINI_API_KEY") else "✗"}

@app.get("/api/models")
async def list_models(category: Optional[str]=None, search: Optional[str]=None, limit:int=50):
    models = get_all_models()
    if category and category!='all': models = [m for m in models if m.category.lower()==category.lower()]
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower() or any(s in t for t in m.tags)]
    return [m.model_dump() for m in models[:limit]]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    for m in get_all_models():
        if m.id == model_id: return m.model_dump()
    raise HTTPException(404, "Not found")

@app.get("/api/categories")
async def get_categories():
    cats = {}
    for m in get_all_models(): cats[m.category] = cats.get(m.category,0)+1
    return {"categories":[{"id":k.lower().replace(" ","-"),"name":k,"count":v} for k,v in cats.items()]}

@app.get("/api/stats")
async def get_stats():
    models = get_all_models()
    return {"total_models":len(models), "total_likes":sum(m.stats.get('likes',0) for m in models), "total_inferences":sum(m.stats.get('inferences',0) for m in models), "categories":len(set(m.category for m in models)), "user_created":len(user_models)}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    model = next((m for m in get_all_models() if m.id==req.model_id), None) if req.model_id else None
    logger.info(f"Chat: model={req.model_id}, query={req.query[:50]}")
    reply = await generate_ai_response(req.query, model)
    if sid not in chat_sessions: chat_sessions[sid] = []
    chat_sessions[sid].append({"role":"user","content":req.query})
    chat_sessions[sid].append({"role":"assistant","content":reply})
    return {"reply":reply, "session_id":sid}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status":"queued","progress":0}
    bg.add_task(process_creation, tid, req)
    return {"task_id":tid, "status":"queued"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t: raise HTTPException(404, "Not found")
    return t

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 OpenGradient v2.2")
    logger.info(f"Gemini: {'✓' if os.getenv('GEMINI_API_KEY') else '✗'}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
