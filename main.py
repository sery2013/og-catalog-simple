from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

# Настройка
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OpenGradient Catalog",
    version="1.0.0",
    docs_url="/docs"
)

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
    stats: Dict[str, int] = {}

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

# === ДАННЫЕ ===

chat_sessions: Dict[str, List] = {}
model_tasks: Dict[str, Dict] = {}

def get_models():
    return [
        ModelInfo(id="og-volatility-eth", name="ETH Volatility Predictor", description="Predicts ETH/USDT volatility using 1h OHLCV data", category="Risk", tags=["defi", "prediction"], stats={"likes": 42, "inferences": 1287}),
        ModelInfo(id="og-llama-finance", name="Llama3 Financial", description="Fine-tuned Llama 3 for financial analysis", category="Language", tags=["llm", "finance"], stats={"likes": 156, "inferences": 5432}),
        ModelInfo(id="og-risk-bert", name="Risk BERT", description="BERT model for DeFi risk scoring", category="Risk", tags=["risk", "defi"], stats={"likes": 89, "inferences": 3421}),
        ModelInfo(id="og-defi-gemma", name="DeFi Gemma", description="Gemma for yield farming strategies", category="DeFi", tags=["defi", "yield"], stats={"likes": 203, "inferences": 8765}),
    ]

# === ROUTES ===

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/models")
async def list_models(category: Optional[str] = None, search: Optional[str] = None):
    models = get_models()
    if category:
        models = [m for m in models if m.category.lower() == category.lower()]
    if search:
        s = search.lower()
        models = [m for m in models if s in m.name.lower() or s in m.description.lower()]
    return [m.model_dump() for m in models]

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    for m in get_models():
        if m.id == model_id:
            return m.model_dump()
    raise HTTPException(404, "Model not found")

@app.get("/api/categories")
async def get_categories():
    return {"categories": [
        {"id": "defi", "name": "DeFi", "count": 128},
        {"id": "language", "name": "Language", "count": 229},
        {"id": "risk", "name": "Risk", "count": 108},
        {"id": "multimodal", "name": "Multimodal", "count": 35},
    ]}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or f"s_{datetime.now().timestamp()}"
    if req.model_id:
        m = next((x for x in get_models() if x.id == req.model_id), None)
        reply = f"📊 **{req.model_id}**\n"
        if m:
            reply += f"• Категория: {m.category}\n• {m.description}\n• Теги: {', '.join(m.tags)}\n• 👍{m.stats.get('likes',0)} 🔄{m.stats.get('inferences',0)}"
        reply += "\n\nЧем помочь?"
    else:
        reply = "👋 Выберите модель из каталога, чтобы задать вопрос!"
    
    if sid not in chat_sessions: chat_sessions[sid] = []
    chat_sessions[sid].append({"role": "user", "content": req.query})
    chat_sessions[sid].append({"role": "assistant", "content": reply})
    return {"reply": reply, "session_id": sid}

@app.post("/api/models/create")
async def create_model(req: CreateModelRequest, bg: BackgroundTasks):
    tid = f"t_{datetime.now().timestamp()}"
    model_tasks[tid] = {"status": "processing", "created": datetime.now().isoformat()}
    
    async def process():
        import asyncio
        await asyncio.sleep(2)
        model_tasks[tid]["status"] = "completed"
        model_tasks[tid]["result"] = {"model_id": f"og-{req.name.lower().replace(' ','-')}"}
    bg.add_task(process)
    return {"task_id": tid, "status": "processing"}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    t = model_tasks.get(task_id)
    if not t: raise HTTPException(404, "Task not found")
    return t

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
