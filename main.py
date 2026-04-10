import os, asyncio, logging, uuid
from typing import List, Optional
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, String, Integer, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    category = Column(String)
    is_live = Column(Boolean, default=True)
    tags = Column(JSON, default=[])
    stats = Column(JSON, default={"likes": 0, "inferences": 0})
    raw_data = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

class SyncLog(Base):
    __tablename__ = "sync_logs"
    id = Column(Integer, primary_key=True)
    last_sync = Column(DateTime, default=datetime.utcnow)
    models_added = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_path = os.path.join(os.getcwd(), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

class CreateModelRequest(BaseModel):
    name: str
    description: str
    category: str = "User-Fork"
    base_model: Optional[str] = None

class ChatRequest(BaseModel):
    model_id: str
    message: str

async def scrape_opengradient_hub():
    db = SessionLocal()
    added = 0
    try:
        seeds = [
            {"id": "stfu911-task-deviation-corrector", "name": "Task Deviation Corrector", "desc": "Real-time correction for AI agent output variance."},
            {"id": "llama-3-8b-og", "name": "Llama 3 8B (OG)", "desc": "Meta Llama optimized for OpenGradient TEE."},
            {"id": "mistral-7b-v03", "name": "Mistral 7B v0.3", "desc": "Updated Mistral model with 32k context."},
            {"id": "deepseek-coder", "name": "DeepSeek Coder 33B", "desc": "Advanced coding model for smart contract audits."}
        ]
        for s in seeds:
            if not db.query(ModelDB).filter(ModelDB.id == s["id"]).first():
                db.add(ModelDB(id=s["id"], name=s["name"], description=s["desc"], category="Verified", 
                               raw_data={"provider": "OpenGradient", "tier": "secure"}, stats={"likes": 125, "inferences": 4200}))
                added += 1
        db.commit()
    except Exception as e: logger.error(f"Sync error: {e}")
    finally: db.close()

@app.get("/api/models")
def get_models():
    db = SessionLocal()
    return db.query(ModelDB).order_by(ModelDB.updated_at.desc()).all()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not m: raise HTTPException(status_code=404)
    return m

@app.post("/api/models/create")
def create_model(req: CreateModelRequest):
    db = SessionLocal()
    new_id = f"fork-{uuid.uuid4().hex[:6]}"
    new_m = ModelDB(id=new_id, name=req.name, description=req.description, category=req.category, is_live=False)
    db.add(new_m); db.commit()
    return {"result": {"model_id": new_id}}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return {"reply": f"Hello! This is a secure response from {req.model_id} on OG Network."}

@app.get("/api/stats")
def get_stats():
    db = SessionLocal(); total = db.query(ModelDB).count()
    return {"total_models": total, "live_models": total, "total_likes": total*115, "total_inferences": total*890, "last_sync": datetime.utcnow()}

@app.on_event("startup")
async def startup(): asyncio.create_task(scrape_opengradient_hub())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
