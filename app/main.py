from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import diabetes, diseases, knowledge, rfp
from app.db.session import init_db
from app.utils.logging import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, knowledge

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize logging, DB and heavy services
    setup_logging()
    print("Initializing Database...")
    init_db()
    
    print("RFPForge API is ready. AI/RAG services will lazy-load when needed.")
    yield
    # Shutdown: Clean up if needed
    print("Shutting down...")

app = FastAPI(title="RFPForge API", lifespan=lifespan)
# CORS (IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register APIs
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(diseases.router)
app.include_router(diabetes.router)
# Include routers
# app.include_router(rfp.router)
# app.include_router(knowledge.router)

@app.get("/")
def root():
    return {"message": "RFPForge API is running"}
