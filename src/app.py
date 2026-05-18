from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.config import BrioConfig
from src.brio_agent import BrioAgent

API_KEY_NAME = "X-API-Key"

config = BrioConfig()
agent: BrioAgent = None  # Initialised in lifespan


class ChatRequest(BaseModel):
    customer_message: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    confidence: float
    escalation_required: bool
    knowledge_used: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate config and initialise the agent on startup (not at import time)."""
    global agent
    config.validate()       # raises clear error if env vars are missing
    agent = BrioAgent(config)
    yield
    # cleanup goes here if needed


app = FastAPI(
    title="BRIO AI Assistant API",
    description="FastAPI wrapper for the BRIO conversational agent.",
    version="1.0.0",
    lifespan=lifespan,
)


def validate_api_key(api_key: str = Header(..., alias=API_KEY_NAME)):
    if api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "BRIO API"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, api_key: str = Depends(validate_api_key)):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    try:
        return agent.respond(request.customer_message)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
