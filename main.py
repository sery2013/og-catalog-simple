import os, asyncio, uuid, random
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import Column, String, Boolean, Text, JSON, DateTime, create_engine, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ModelDB(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(Text)
    category = Column(String)
    is_live = Column(Boolean, default=True)
    type = Column(String, default="BASE")
    stats = Column(JSON)
    raw_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
app = FastAPI()

static_path = os.path.join(os.getcwd(), 'static')
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))

@app.get("/api/models")
def get_models():
    db = SessionLocal()
    # Сортировка по дате создания (новые вверху)
    return db.query(ModelDB).order_by(desc(ModelDB.created_at)).all()

@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    db = SessionLocal()
    m = db.query(ModelDB).filter(ModelDB.id == model_id).first()
    if not m: raise HTTPException(status_code=404)
    return m

class CreateModelRequest(BaseModel):
    name: str
    description: str

@app.post("/api/models/create")
def create_model(req: CreateModelRequest):
    db = SessionLocal()
    new_id = f"custom-{uuid.uuid4().hex[:6]}"
    new_m = ModelDB(
        id=new_id,
        name=req.name,
        description=req.description,
        category="User Model",
        type="USER",
        stats={"likes": 0, "inferences": 0},
        raw_data={"status": "deployed", "deployment_id": str(uuid.uuid4())},
        created_at=datetime.utcnow()
    )
    db.add(new_m)
    db.commit()
    return {"status": "success"}

@app.post("/api/chat")
async def chat(req: dict):
    return {"reply": f"Response from {req['model_id']}: Analysis complete via SDK."}

@app.on_event("startup")
async def seed():
    db = SessionLocal()
    if db.query(ModelDB).count() == 0:
        for i in range(5):
            db.add(ModelDB(
                id=f"model-{i}", name=f"Model Alpha {i}", 
                description="Seeded model for testing", category="LLM",
                stats={"likes": random.randint(10, 1000), "inferences": random.randint(100, 5000)},
                raw_data={"version": "1.0", "engine": "OG-v1"}
            ))
        db.commit()
