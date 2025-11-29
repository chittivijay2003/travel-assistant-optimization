"""
FastAPI Main Application
Enterprise Travel Assistant with comprehensive features
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project components
from travel_assistant.core.config import settings
from travel_assistant.core.logger import logger

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime


# Request/Response models
class TravelQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(default="anonymous")
    include_model_comparison: bool = Field(default=False)
    use_cache: bool = Field(default=True)


class TravelQueryResponse(BaseModel):
    query: str
    response: str
    user_id: str
    metrics: Dict[str, Any]
    processing_time_ms: float
    timestamp: str
    success: bool
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for FastAPI app"""
    logger.info("🚀 Starting Enterprise Travel Assistant API")
    yield
    logger.info("🛑 Shutting down Enterprise Travel Assistant API")


# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Travel Assistant API",
    description="AI-powered travel assistant with memory, caching, and fingerprinting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🧳 Enterprise Travel Assistant API",
        "version": "1.0.0",
        "features": [
            "Memory management with Mem0",
            "Semantic caching with Redis",
            "Request fingerprinting",
            "Model comparison (Gemini Flash vs Pro)",
            "LangGraph workflow orchestration",
        ],
        "endpoints": {
            "docs": "/docs",
            "chat": "/chat",
            "health": "/health",
            "metrics": "/metrics",
            "assistant": "/memory-travel-assistant",
        },
    }


@app.post("/memory-travel-assistant", response_model=TravelQueryResponse)
async def process_travel_query(request: TravelQueryRequest):
    """Main travel assistant endpoint"""
    try:
        # For demonstration - would integrate with actual workflow
        response_text = f"""
I understand you're asking about: "{request.query}"

As your AI travel assistant, I'm here to help with all your travel planning needs! 
While I'm currently in demo mode, in production I would:

🧠 Remember your preferences using Mem0 memory system
🗄️ Check for similar queries in my semantic cache
🔑 Generate a unique fingerprint for your request
⚡ Compare responses from Gemini Flash and Pro models
🔄 Use LangGraph to orchestrate the entire workflow

Your query would be processed through our enterprise pipeline with comprehensive 
error handling and performance metrics.

How can I help make your travel dreams come true?
        """

        # Demo metrics
        demo_metrics = {
            "response_source": "demo_mode",
            "cache_info": {"cache_hit": False, "similarity_score": 0.0},
            "memory_info": {"memories_found": 0, "context_retrieved": False},
            "fingerprint_info": {
                "is_duplicate": False,
                "fingerprint_hash": "demo_hash",
            },
            "model_comparison": {
                "flash_time_ms": 150,
                "pro_time_ms": 300,
                "speed_winner": "gemini-flash",
            },
        }

        return TravelQueryResponse(
            query=request.query,
            response=response_text,
            user_id=request.user_id,
            metrics=demo_metrics,
            processing_time_ms=200.0,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
        )

    except Exception as e:
        logger.error(f"❌ Error processing travel query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": settings.environment,
        "components": {"api": "healthy", "configuration": "loaded"},
    }


@app.get("/metrics")
async def get_metrics():
    """Metrics endpoint"""
    return {
        "system_metrics": {
            "uptime_seconds": 0,
            "total_requests": 0,
            "api_version": "1.0.0",
            "environment": settings.environment,
        },
        "demo_metrics": {
            "note": "This is demo mode - full metrics available when components are integrated"
        },
    }


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Web-based chat interface"""
    return HTMLResponse(
        content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Travel Assistant Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { background: linear-gradient(90deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 10px 10px 0 0; text-align: center; }
            .chat { height: 400px; overflow-y: auto; padding: 20px; border-bottom: 1px solid #ddd; }
            .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
            .user { background: #e3f2fd; margin-left: 50px; }
            .bot { background: #f5f5f5; margin-right: 50px; }
            .input-area { padding: 20px; display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧳 Enterprise Travel Assistant</h1>
                <p>AI-powered travel planning with memory & caching</p>
            </div>
            <div class="chat" id="chat">
                <div class="message bot">
                    <strong>🤖 Travel Assistant:</strong><br>
                    Welcome! I'm your AI travel assistant. I can help you plan trips, find destinations, 
                    and provide personalized recommendations. What travel adventure are you planning?
                </div>
            </div>
            <div class="input-area">
                <input type="text" id="input" placeholder="Ask me about travel..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
            async function sendMessage() {
                const input = document.getElementById('input');
                const chat = document.getElementById('chat');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message
                chat.innerHTML += `<div class="message user"><strong>👤 You:</strong><br>${message}</div>`;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;
                
                // Add loading
                chat.innerHTML += '<div class="message bot" id="loading"><strong>🤖 Travel Assistant:</strong><br>Thinking...</div>';
                chat.scrollTop = chat.scrollHeight;
                
                try {
                    const response = await fetch('/memory-travel-assistant', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: message, user_id: 'web_user'})
                    });
                    
                    const data = await response.json();
                    
                    // Remove loading
                    document.getElementById('loading').remove();
                    
                    // Add response
                    chat.innerHTML += `<div class="message bot"><strong>🤖 Travel Assistant:</strong><br>${data.response}</div>`;
                    
                } catch (error) {
                    document.getElementById('loading').innerHTML = '<strong>🤖 Travel Assistant:</strong><br>Sorry, I encountered an error. Please try again.';
                }
                
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    )


def main():
    """Main entry point for the application"""
    logger.info("🧳 Starting Travel Assistant Application")
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"⚙️ Environment: {settings.environment}")
    logger.info("✅ FastAPI application configured")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
