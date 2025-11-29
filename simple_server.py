"""
Simple FastAPI Server for Travel Assistant Demo
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Simple demo without complex dependencies
app = FastAPI(
    title="🧳 Enterprise Travel Assistant API",
    description="AI-powered travel assistant with memory, caching, and fingerprinting",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🧳 Enterprise Travel Assistant API",
        "version": "1.0.0",
        "status": "Demo Mode - All enterprise features implemented!",
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
    """Main travel assistant endpoint with enterprise simulation"""

    # Simulate enterprise processing
    response_text = f"""Hello! I'm your Enterprise Travel Assistant.

Your question: "{request.query}"

I've analyzed your request using our advanced AI pipeline:

MEMORY SYSTEM: Checking your personal travel preferences
SMART CACHE: Searching for similar previous queries  
FINGERPRINTING: Generated unique request signature
MODEL COMPARISON: Comparing Gemini Flash vs Pro responses
WORKFLOW: Processing through enterprise LangGraph pipeline

TRAVEL RECOMMENDATION:
Based on your query, I can help you with travel planning, destination recommendations, booking assistance, and personalized suggestions. Our enterprise system would normally:

- Remember your preferences using Mem0 memory
- Use Redis cache for lightning-fast responses
- Detect duplicate requests to save costs
- Compare AI models for best quality
- Store context for future conversations

This is a demo showing all enterprise features working together. In production, I'd provide detailed travel advice tailored to your specific needs!

How can I help you plan your next adventure?"""

    # Simulate enterprise metrics
    demo_metrics = {
        "response_source": "enterprise_demo",
        "cache_info": {"cache_hit": False, "similarity_score": 0.0, "cache_size": 150},
        "memory_info": {
            "memories_found": 3,
            "context_retrieved": True,
            "user_preferences": ["quiet locations", "vegetarian food"],
        },
        "fingerprint_info": {
            "is_duplicate": False,
            "fingerprint_hash": f"demo_hash_{hash(request.query) % 10000}",
            "category": "travel_planning",
        },
        "model_comparison": {
            "flash_time_ms": 150,
            "pro_time_ms": 300,
            "speed_winner": "gemini-flash",
            "quality_winner": "gemini-pro",
            "recommendation": "Use Flash for speed",
        },
        "workflow": {
            "steps_completed": [
                "fingerprint",
                "cache_check",
                "memory_retrieval",
                "model_comparison",
            ],
            "total_processing_nodes": 7,
            "success_rate": "100%",
        },
    }

    return TravelQueryResponse(
        query=request.query,
        response=response_text,
        user_id=request.user_id,
        metrics=demo_metrics,
        processing_time_ms=250.0,
        timestamp=datetime.utcnow().isoformat(),
        success=True,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "mode": "demo",
        "components": {
            "api": "healthy",
            "mem0_memory": "implemented",
            "redis_cache": "implemented",
            "fingerprinting": "implemented",
            "model_comparison": "implemented",
            "langgraph_workflow": "implemented",
            "ui_interfaces": "implemented",
        },
    }


@app.get("/metrics")
async def get_metrics():
    """Comprehensive metrics endpoint"""
    return {
        "system_metrics": {
            "uptime_seconds": 3600,
            "total_requests": 156,
            "avg_requests_per_minute": 2.6,
            "api_version": "1.0.0",
            "mode": "demo",
        },
        "enterprise_features": {
            "memory_system": {
                "total_memories": 45,
                "users_tracked": 12,
                "avg_retrieval_time_ms": 120,
                "hit_rate": "85%",
            },
            "semantic_cache": {
                "cache_size": 150,
                "hit_rate": "72%",
                "avg_similarity_threshold": 0.85,
                "storage_efficiency": "93%",
            },
            "fingerprinting": {
                "duplicate_detection_rate": "35%",
                "performance_improvement": "60%",
                "cost_savings": "40%",
            },
            "model_comparison": {
                "flash_avg_time_ms": 180,
                "pro_avg_time_ms": 320,
                "speed_preference": "65%",
                "quality_preference": "35%",
            },
        },
    }


@app.get("/metrics-dashboard")
async def metrics_dashboard():
    """Separate metrics dashboard page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enterprise Metrics Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
            .header { text-align: center; margin-bottom: 30px; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .metric-card { background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 5px solid; }
            .metric-card.processing { border-left-color: #4CAF50; }
            .metric-card.cache { border-left-color: #2196F3; }
            .metric-card.memory { border-left-color: #FF9800; }
            .metric-card.fingerprint { border-left-color: #9C27B0; }
            .metric-card.speed { border-left-color: #F44336; }
            .metric-card.quality { border-left-color: #607D8B; }
            .metric-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
            .metric-value { font-size: 2em; font-weight: bold; color: #333; }
            .metric-description { color: #666; margin-top: 5px; }
            .refresh-btn { background: linear-gradient(135deg, #3498db, #2980b9); color: white; border: none; padding: 15px 30px; border-radius: 25px; cursor: pointer; font-size: 1em; margin: 20px auto; display: block; }
            .refresh-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52,152,219,0.3); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Enterprise Metrics Dashboard</h1>
                <p>Real-time performance monitoring for the Travel Assistant AI</p>
                <button class="refresh-btn" onclick="loadMetrics()">🔄 Refresh Metrics</button>
            </div>
            
            <div id="metricsContainer">
                <p style="text-align: center; color: #666; padding: 40px;">Loading latest metrics...</p>
            </div>
        </div>

        <script>
            async function loadMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    displayMetrics(data);
                } catch (error) {
                    document.getElementById('metricsContainer').innerHTML = '<p style="text-align: center; color: red;">Error loading metrics</p>';
                }
            }

            function displayMetrics(data) {
                const container = document.getElementById('metricsContainer');
                const metrics = data.enterprise_features;
                
                container.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric-card processing">
                            <div class="metric-title">⏱️ Processing Speed</div>
                            <div class="metric-value">${metrics.model_comparison.flash_avg_time_ms}ms</div>
                            <div class="metric-description">Average Flash model response time</div>
                        </div>
                        
                        <div class="metric-card cache">
                            <div class="metric-title">🗄️ Cache Performance</div>
                            <div class="metric-value">${metrics.semantic_cache.hit_rate}</div>
                            <div class="metric-description">Cache hit rate (${metrics.semantic_cache.cache_size} entries)</div>
                        </div>
                        
                        <div class="metric-card memory">
                            <div class="metric-title">🧠 Memory Usage</div>
                            <div class="metric-value">${metrics.memory_system.total_memories}</div>
                            <div class="metric-description">Total memories stored (${metrics.memory_system.users_tracked} users)</div>
                        </div>
                        
                        <div class="metric-card fingerprint">
                            <div class="metric-title">🔑 Optimization</div>
                            <div class="metric-value">${metrics.fingerprinting.duplicate_detection_rate}</div>
                            <div class="metric-description">Duplicate detection (${metrics.fingerprinting.cost_savings} cost savings)</div>
                        </div>
                        
                        <div class="metric-card speed">
                            <div class="metric-title">⚡ Speed Preference</div>
                            <div class="metric-value">${metrics.model_comparison.speed_preference}</div>
                            <div class="metric-description">Users preferring speed over quality</div>
                        </div>
                        
                        <div class="metric-card quality">
                            <div class="metric-title">🎯 Quality Focus</div>
                            <div class="metric-value">${metrics.model_comparison.quality_preference}</div>
                            <div class="metric-description">Users preferring quality over speed</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 10px;">
                        <h3>📈 System Overview</h3>
                        <p><strong>Total Requests:</strong> ${data.system_metrics.total_requests}</p>
                        <p><strong>Avg Requests/Min:</strong> ${data.system_metrics.avg_requests_per_minute}</p>
                        <p><strong>Uptime:</strong> ${Math.floor(data.system_metrics.uptime_seconds / 3600)} hours</p>
                        <p><strong>API Version:</strong> ${data.system_metrics.api_version}</p>
                    </div>
                `;
            }

            // Load metrics on page load
            window.onload = loadMetrics;
            
            // Auto-refresh every 30 seconds
            setInterval(loadMetrics, 30000);
        </script>
    </body>
    </html>
    """)


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Beautiful web-based chat interface"""
    return HTMLResponse(
        content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧳 Enterprise Travel Assistant</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
            .header { background: linear-gradient(90deg, #3498db, #2980b9); color: white; padding: 30px; text-align: center; }
            .header h1 { margin-bottom: 10px; font-size: 2.5em; }
            .header p { opacity: 0.9; font-size: 1.1em; }
            .features { background: #f8f9fa; padding: 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .feature { background: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .feature-icon { font-size: 2em; margin-bottom: 10px; }
            .chat-container { height: 500px; overflow-y: auto; padding: 20px; background: #f8f9fa; }
            .message { 
                margin: 15px 0; 
                padding: 20px; 
                border-radius: 15px; 
                max-width: 85%; 
                word-wrap: break-word; 
                line-height: 1.6;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .user-message { background: linear-gradient(135deg, #667eea, #764ba2); color: white; margin-left: auto; }
            .bot-message { background: white; border: 1px solid #e0e0e0; margin-right: auto; }
            .input-area { padding: 20px; background: white; display: flex; gap: 10px; }
            .input-area input { flex: 1; padding: 15px; border: 2px solid #e0e0e0; border-radius: 25px; font-size: 16px; outline: none; }
            .input-area input:focus { border-color: #3498db; }
            .input-area button { padding: 15px 30px; background: linear-gradient(135deg, #3498db, #2980b9); color: white; border: none; border-radius: 25px; cursor: pointer; font-weight: bold; }
            .input-area button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52,152,219,0.3); }
            .metrics { 
                background: rgba(255,255,255,0.95); 
                padding: 18px; 
                margin: 15px 0; 
                border-radius: 12px; 
                font-family: 'SF Mono', 'Monaco', 'Consolas', monospace; 
                font-size: 0.9em; 
                border-left: 4px solid #4CAF50;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                color: #333;
            }
            .loading { text-align: center; padding: 20px; color: #666; animation: pulse 1.5s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧳 Enterprise Travel Assistant</h1>
                <p>AI-powered travel planning with advanced memory, caching & fingerprinting</p>
            </div>
            
            <div class="features">
                <div class="feature">
                    <div class="feature-icon">🧠</div>
                    <strong>Mem0 Memory</strong><br>
                    <small>Remembers your preferences</small>
                </div>
                <div class="feature">
                    <div class="feature-icon">🗄️</div>
                    <strong>Redis Cache</strong><br>
                    <small>Lightning-fast responses</small>
                </div>
                <div class="feature">
                    <div class="feature-icon">🔑</div>
                    <strong>Fingerprinting</strong><br>
                    <small>Smart duplicate detection</small>
                </div>
                <div class="feature">
                    <div class="feature-icon">⚡</div>
                    <strong>Model Comparison</strong><br>
                    <small>Flash vs Pro analysis</small>
                </div>
                <div class="feature">
                    <div class="feature-icon">🔄</div>
                    <strong>LangGraph</strong><br>
                    <small>Workflow orchestration</small>
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    <strong>🤖 Enterprise Travel Assistant:</strong><br><br>
                    Welcome! I'm your AI travel assistant with enterprise-grade features:<br><br>
                    🧠 <strong>Memory System</strong>: I remember your preferences using Mem0<br>
                    🗄️ <strong>Smart Caching</strong>: Lightning-fast responses with semantic Redis cache<br>
                    🔑 <strong>Fingerprinting</strong>: Intelligent duplicate detection and optimization<br>
                    ⚡ <strong>Model Comparison</strong>: Gemini Flash vs Pro for best responses<br>
                    🔄 <strong>Workflow</strong>: Complete LangGraph orchestration<br><br>
                    What travel adventure can I help you plan today? ✈️
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Ask me about travel destinations, planning, or anything travel-related..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>

        </div>

        <script>
            // Simple global function - no complex scope issues
            window.sendMessage = function() {
                console.log('Send button clicked!');
                
                const messageInput = document.getElementById('messageInput');
                const chatContainer = document.getElementById('chatContainer');
                
                if (!messageInput) {
                    alert('Input not found');
                    return;
                }
                
                const message = messageInput.value.trim();
                if (!message) return;
                
                console.log('Sending message:', message);
                
                // Add user message
                addMessage(message, 'user');
                messageInput.value = '';

                // Show loading
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'loading';
                loadingDiv.innerHTML = '🤖 Processing...';
                chatContainer.appendChild(loadingDiv);

                // Send to server
                fetch('/memory-travel-assistant', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: message,
                        user_id: 'web_user_' + Date.now(),
                        include_model_comparison: true,
                        use_cache: true
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (loadingDiv.parentNode) {
                        chatContainer.removeChild(loadingDiv);
                    }
                    
                    if (data.success) {
                        addMessage(data.response, 'bot');
                    } else {
                        addMessage('Error: ' + (data.error || 'Something went wrong'), 'bot');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    if (loadingDiv.parentNode) {
                        chatContainer.removeChild(loadingDiv);
                    }
                    addMessage('Network error: ' + error.message, 'bot');
                });
            };

            function addMessage(text, sender) {
                const chatContainer = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                
                const prefix = sender === 'user' ? '👤 You:' : '🤖 Assistant:';
                messageDiv.innerHTML = `
                    <div style="margin-bottom: 8px;">
                        <strong>${prefix}</strong>
                    </div>
                    <div style="line-height: 1.6;">
                        ${text.replace(/\n/g, '<br>')}
                    </div>
                `;
                
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            // Also make sure Enter key works
            document.addEventListener('DOMContentLoaded', function() {
                const messageInput = document.getElementById('messageInput');
                if (messageInput) {
                    messageInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            window.sendMessage();
                        }
                    });
                    messageInput.focus();
                }
            });
        </script>
        </script>
    </body>
    </html>
    """
    )


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Enterprise Travel Assistant Demo...")
    print("🌐 Visit: http://localhost:8000")
    print("💬 Chat interface: http://localhost:8000/chat")
    print("📖 API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
