from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from wms import load_capabilities, get_all_layers
from agent import chat, create_session


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_capabilities()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ActiveLayer(BaseModel):
    name: str
    title: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    active_layers: list[ActiveLayer] = []


class ChatResponse(BaseModel):
    reply: str
    actions: list[dict]


@app.post("/api/session")
async def api_create_session():
    return {"session_id": create_session()}


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> ChatResponse:
    try:
        layers = [{"name": layers.name, "title": layers.title} for layers in req.active_layers
        ]
        reply, actions = await chat(req.session_id, req.message, layers)
    except KeyError:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")
    return ChatResponse(reply=reply, actions=actions)


@app.get("/api/layers")
async def api_layers():
    return get_all_layers()


# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
