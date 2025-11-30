# 🧳 Travel Assistant - AI-Powered Travel Recommendation System

## Overview

An intelligent travel recommendation system powered by Google Gemini AI that learns from user preferences and provides personalized travel suggestions. The system combines advanced AI models with semantic caching and memory management to deliver fast, contextual responses.

### Key Features

- **Memory Management** - Learns and remembers user preferences across sessions
- **Semantic Caching** - Intelligent response caching using embeddings for faster responses
- **Request Fingerprinting** - Efficient duplicate detection and request tracking
- **Dual Model Comparison** - Gemini Flash (speed) and Pro (quality) model comparison
- **LangGraph Workflow** - Sophisticated multi-node workflow orchestration
- **RESTful API** - FastAPI-based API with comprehensive documentation
- **Location-Aware** - Intelligently handles multi-location conversations

## 🚀 Quick Start

### 1. Get Your API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

### 2. Configure Environment
Open `.env` file and add your API key:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python3 main.py
```

The server will start on http://localhost:8001

## 📡 API Endpoints

### Main Endpoint
**POST** `/memory-travel-assistant`

Request:
```json
{
  "query": "Recommend quiet beach destinations for vegetarians",
  "user_id": "user123",
  "include_model_comparison": false
}
```

Response:
```json
{
  "query": "Recommend quiet beach destinations for vegetarians",
  "response": "Based on your preferences... [AI-generated response]",
  "user_id": "user123",
  "metadata": {
    "source": "ai_generated",
    "model": "gemini-flash",
    "latency_ms": 1234.56,
    "has_memory_context": true
  },
  "timestamp": "2025-11-29T12:00:00"
}
```

### Other Endpoints
- **GET** `/` - Service information
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation

## 🧪 Testing

### Using curl:
```bash
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Plan a beach vacation",
    "user_id": "test_user"
  }'
```

### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8001/memory-travel-assistant",
    json={
        "query": "Recommend beach destinations for vegetarians",
        "user_id": "user123"
    }
)

print(response.json())
```

### Using Browser:
Visit http://localhost:8001/docs for interactive testing

## 🏗️ Architecture

### Workflow Flow:
```
Query → Fingerprinting → Cache Check → Memory Retrieval → AI Generation → Memory Update → Response
                              ↓
                          Cache Hit?
                              ↓
                          Return Cached
```

### Components:

1. **MemoryManager** - Stores and retrieves user preferences using Mem0
2. **SemanticCache** - Caches responses with sentence embeddings
3. **RequestFingerprinter** - SHA-256 fingerprints for duplicate detection
4. **GeminiModelComparator** - Compares Flash vs Pro models
5. **TravelAssistantWorkflow** - LangGraph orchestration
6. **FastAPI App** - REST API server

## 📁 Project Structure

```
travel-assistant-optimization/
├── main.py                 # Application entry point
├── travel_assistant.py     # Core implementation
├── travel_assistant.ipynb  # Jupyter notebook demo
├── .env                    # Configuration (add your API key here)
├── .env.example           # Example configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## ⚙️ Configuration

Edit `.env` file:

```bash
# REQUIRED
GOOGLE_API_KEY=your_key_here

# Optional
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600
CACHE_THRESHOLD=0.85
PORT=8001
```

## 🔧 Features

### Memory Management
- Stores user preferences persistently
- Retrieves relevant context for queries
- Updates after each conversation
- Fallback storage when Mem0 unavailable

### Semantic Caching
- Caches AI responses with embeddings
- Retrieves similar queries (cosine similarity > 0.85)
- TTL-based expiration (1 hour default)
- Fallback cache when Redis unavailable

### Model Comparison
- **Gemini Flash**: Faster responses (~150ms)
- **Gemini Pro**: More detailed responses (~450ms)
- Tracks latency, length, quality metrics

### Request Fingerprinting
- SHA-256 hashing of queries
- Duplicate detection per user per day
- Request counting and tracking

## 🎯 Technical Capabilities

✅ **Memory Management** - Mem0 integration with fallback storage  
✅ **Semantic Caching** - Redis-based cache with sentence embeddings  
✅ **Request Fingerprinting** - SHA-256 hashing for duplicate detection  
✅ **Model Comparison** - Gemini Flash vs Pro performance analysis  
✅ **Workflow Orchestration** - LangGraph multi-node workflow  
✅ **API Endpoints** - Production-ready FastAPI implementation  
✅ **Location Context** - Smart location extraction and context management

## 🐛 Troubleshooting

### "API key not configured"
- Make sure you added your key to `.env` file
- Check the key is on the line: `GOOGLE_API_KEY=your_key`
- No quotes or spaces around the key

### "Redis connection failed"
- Normal! The app will use fallback cache
- Optional: Install Redis for better caching

### "Mem0 unavailable"
- Normal! The app will use fallback storage
- Optional: Configure Mem0 API key

## 📚 Documentation

- **API Docs**: http://localhost:8001/docs (when server is running)
- **Gemini API**: https://ai.google.dev/
- **LangGraph**: https://python.langchain.com/docs/langgraph

## 🤝 Support

For issues or questions:
1. Check `.env` has your API key
2. Verify Python 3.8+ is installed
3. Check all dependencies are installed
4. Review server logs for errors

## 🌟 Use Cases

- **Personalized Travel Planning** - Get recommendations based on your preferences
- **Multi-City Itineraries** - Plan trips across multiple destinations
- **Budget Travel** - Find options within your budget constraints
- **Dietary Preferences** - Vegetarian, vegan, or specific cuisine recommendations
- **Activity-Based Travel** - Adventure, relaxation, culture, or family-friendly options
- **Follow-Up Queries** - Natural conversation flow with context awareness

## 🔒 Security & Privacy

- User data isolated by `user_id`
- API key stored securely in environment variables
- No data sharing between users
- In-memory fallback when external services unavailable

## 📈 Performance

- **Gemini Flash**: ~2-20s response time (optimized for speed)
- **Gemini Pro**: ~12-35s response time (optimized for quality)
- **Cache Hit**: < 1s response time
- **Semantic Similarity**: 0.85 threshold for cache matching
- **Concurrent Requests**: Supports multiple users simultaneously

---

## 🔗 Repository

**GitHub**: [https://github.com/chittivijay2003/travel-assistant-optimization](https://github.com/chittivijay2003/travel-assistant-optimization)

---

**Built with Google Gemini AI, LangGraph, FastAPI, and Mem0**
