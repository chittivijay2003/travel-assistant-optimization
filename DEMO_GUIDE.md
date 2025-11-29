# 🧳 Enterprise Travel Assistant - Demo & Production Guide

## 🚀 Quick Start Guide

### 1. Start the API Server
```bash
# Option 1: Using the runner script
python run_server.py

# Option 2: Direct uvicorn command
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Dashboard (Optional)
```bash
# In a new terminal
python run_dashboard.py

# Or direct streamlit command
streamlit run travel_assistant/ui/streamlit_app.py --server.port 8501
```

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |
| **API Alternative Docs** | http://localhost:8000/redoc | ReDoc documentation |
| **Web Chat Interface** | http://localhost:8000/chat | Built-in chat interface |
| **Health Check** | http://localhost:8000/health | System status |
| **Metrics** | http://localhost:8000/metrics | Performance metrics |
| **Streamlit Dashboard** | http://localhost:8501 | Full-featured dashboard |

## 🧪 Testing the API

### Using the Web Chat Interface
1. Visit http://localhost:8000/chat
2. Type your travel questions and see responses
3. View processing metrics for each query

### Using curl
```bash
# Basic travel query
curl -X POST "http://localhost:8000/memory-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to plan a romantic getaway to Paris",
    "user_id": "test_user",
    "include_model_comparison": true,
    "use_cache": true
  }'

# Health check
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

### Using Python requests
```python
import requests

# Send a travel query
response = requests.post("http://localhost:8000/memory-travel-assistant", json={
    "query": "Best time to visit Japan for cherry blossoms?",
    "user_id": "python_user",
    "include_model_comparison": True,
    "use_cache": True
})

print(response.json())
```

## 🏗️ Architecture Overview

### Current Demo Implementation
- **FastAPI Server**: Provides REST API endpoints
- **Demo Mode**: Returns simulated responses with realistic metrics
- **Web Chat Interface**: Built-in HTML/JavaScript chat interface
- **Streamlit Dashboard**: Advanced metrics and testing interface

### Enterprise Components (Notebook Implementation)
The Jupyter notebook `travel_assistant_assignment.ipynb` contains full enterprise implementations:

1. **EnterpriseMemoryManager** - Mem0 integration with persistent storage
2. **EnterpriseSemanticCache** - Redis-based semantic caching with similarity matching
3. **EnterpriseRequestFingerprinter** - Duplicate detection and request optimization
4. **EnterpriseModelComparator** - Gemini Flash vs Pro performance analysis
5. **EnterpriseTravelAssistantWorkflow** - LangGraph orchestration pipeline

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
# Required for full functionality
GOOGLE_API_KEY=your_google_api_key_here
REDIS_HOST=localhost
REDIS_PORT=6379
MEM0_API_KEY=your_mem0_api_key_here

# Optional
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Dependencies
All dependencies are defined in `pyproject.toml`:
- FastAPI & Uvicorn for web framework
- Streamlit & Plotly for dashboard
- Google GenerativeAI for LLM integration
- Redis & Mem0 for memory/caching
- LangChain & LangGraph for workflow orchestration

## 📊 Features Demonstration

### Memory System
- User preferences are remembered across sessions
- Contextual information is maintained
- Fallback storage when external services are unavailable

### Semantic Cache
- Similar queries return cached responses instantly
- Configurable similarity thresholds
- Redis-based persistence with TTL management

### Request Fingerprinting
- Duplicate detection prevents redundant processing
- Intelligent request categorization
- Performance optimization analytics

### Model Comparison
- Side-by-side Gemini Flash vs Pro performance
- Response time and quality metrics
- Automated winner selection based on criteria

### LangGraph Workflow
- Orchestrated processing pipeline
- State management throughout the workflow
- Conditional routing and error recovery

## 🎯 Demo Scenarios

### Scenario 1: Travel Planning
```
Query: "I'm planning a family vacation to Europe in July with kids aged 8 and 12"
```
- System remembers family composition and preferences
- Checks cache for similar European family vacation queries
- Compares model responses for comprehensive advice
- Stores new preferences for future queries

### Scenario 2: Destination Research
```
Query: "What's the weather like in Tokyo in March?"
```
- Quick cache lookup for weather information
- Fingerprint analysis to detect if recently asked
- Fast response from Gemini Flash for factual queries

### Scenario 3: Complex Itinerary
```
Query: "Create a 10-day itinerary for Japan including cultural sites and food experiences"
```
- Memory retrieval for past Japan/cultural preferences
- Model comparison to get comprehensive vs concise itineraries
- Cache results for similar detailed planning requests

## 🔍 Monitoring & Metrics

### Real-time Metrics
- Processing times per component
- Cache hit/miss ratios
- Model performance comparisons
- Memory system utilization
- Request fingerprint analytics

### Performance Tracking
- Response time trends
- Success/failure rates
- Component health status
- Resource utilization

## 🚦 Production Considerations

### Security
- API key management through environment variables
- Input validation and sanitization
- Rate limiting and request throttling
- CORS configuration for production domains

### Scalability
- Redis clustering for cache scalability
- Load balancing for multiple API instances
- Background task processing for cleanup
- Metrics collection and alerting

### Reliability
- Graceful degradation when services are unavailable
- Comprehensive error handling and logging
- Health checks for all dependencies
- Retry mechanisms with exponential backoff

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find and kill process using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

2. **Module import errors**
   ```bash
   # Ensure dependencies are installed
   pip install -e .
   # Or using UV
   uv pip sync
   ```

3. **API key errors**
   - Verify `.env` file exists and contains valid keys
   - Check environment variable loading

4. **Redis connection issues**
   - Ensure Redis is running locally or update connection settings
   - System gracefully falls back to in-memory cache

### Debug Mode
Enable debug mode by setting `DEBUG=true` in your `.env` file for detailed logging.

## 📝 Next Steps

To convert this demo into a production system:

1. **API Keys Setup**: Configure actual Google AI and Mem0 API keys
2. **Redis Deployment**: Set up Redis instance (local or cloud)
3. **Model Integration**: Connect to actual Gemini models
4. **Database Setup**: Configure persistent storage for memory system
5. **Testing**: Run comprehensive integration tests
6. **Deployment**: Configure for your target environment

The enterprise implementation in the Jupyter notebook provides the complete foundation for this production deployment.