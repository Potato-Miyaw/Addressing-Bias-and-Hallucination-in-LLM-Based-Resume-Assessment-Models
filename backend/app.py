"""
DSA 9 MVP - FastAPI Backend (Clean Router Structure)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import jd_router, resume_router, verification_router, matching_router, ranking_router, pipeline_router

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import routers
from backend.routers import jd_router, resume_router, verification_router, matching_router, ranking_router, pipeline_router, audit_router

# Initialize FastAPI
app = FastAPI(
    title="DSA 9 MVP - LLM Hiring System API",
    description="Bias-aware, hallucination-detecting resume screening system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jd_router.router)
app.include_router(resume_router.router)
app.include_router(verification_router.router)
app.include_router(matching_router.router)
app.include_router(ranking_router.router)
app.include_router(pipeline_router.router)
app.include_router(audit_router.router)

# Health check
@app.get("/")
async def root():
    """API root - health check"""
    return {
        "service": "DSA 9 MVP - LLM Hiring System",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "endpoints": {
            "jd": "/api/jd/*",
            "resume": "/api/resume/*",
            "verify": "/api/verify/*",
            "match": "/api/match/*",
            "rank": "/api/rank/*"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000
    )
