# 🧳 Enterprise Travel Assistant

A sophisticated AI-powered travel planning system featuring intelligent memory management, semantic caching, and multi-model AI comparison capabilities.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/license-Enterprise-blue.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)

## 🌟 Features

### 🧠 **Intelligent Memory Management**
- **Mem0 Integration**: Persistent user preferences and context across sessions
- **Contextual Learning**: Remembers travel patterns, preferences, and past interactions
- **Dynamic Context Retrieval**: Intelligent memory search for relevant travel history

### 🗄️ **High-Performance Semantic Caching**
- **Redis-Powered**: Enterprise-grade caching with Redis backend
- **Semantic Similarity**: Content-aware caching using sentence transformers
- **Configurable TTL**: Flexible cache expiration policies
- **Cache Analytics**: Performance metrics and hit rate optimization

### 🔑 **Advanced Request Fingerprinting**
- **Deduplication**: Intelligent detection of repeated or similar requests
- **SHA-256 Hashing**: Secure request fingerprint generation
- **Optimization**: Reduces redundant processing and improves performance
- **Analytics Integration**: Request pattern analysis and insights

### 🤖 **Multi-Model AI Comparison**
- **Gemini Flash vs Pro**: Side-by-side model performance analysis
- **Response Quality Metrics**: Automated evaluation of AI outputs
- **Latency Comparison**: Performance benchmarking across models
- **Token Usage Tracking**: Cost optimization insights

### 🔄 **LangGraph Workflow Orchestration**
- **Intelligent Routing**: Dynamic workflow management based on request type
- **State Management**: Persistent conversation state across interactions
- **Error Handling**: Robust fallback mechanisms and recovery
- **Monitoring Integration**: Real-time workflow performance tracking

### 🌐 **Enterprise API & Infrastructure**
- **FastAPI Framework**: High-performance async web framework
- **Auto-Documentation**: Interactive API docs with OpenAPI/Swagger
- **CORS Support**: Cross-origin resource sharing for web applications
- **Health Monitoring**: Comprehensive health checks and status endpoints

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Redis server (optional, fallback available)
- Google AI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/enterprise-travel-assistant.git
cd enterprise-travel-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
GOOGLE_API_KEY=your_google_ai_api_key_here
AI_MODEL=gemini-1.5-flash-latest
REDIS_HOST=localhost
REDIS_PORT=6379
ENVIRONMENT=production
```

4. **Start the application**
```bash
python main.py
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information and capabilities |
| `/memory-travel-assistant` | POST | Main travel assistance endpoint |
| `/health` | GET | Health check and system status |
| `/metrics` | GET | Performance metrics and analytics |
| `/chat` | GET | Web-based chat interface |
| `/docs` | GET | Interactive API documentation |

### Travel Assistant Endpoint

**POST /memory-travel-assistant**

Request body:
```json
{
  "query": "Plan a 5-day trip to Japan in spring",
  "user_id": "user_123",
  "include_model_comparison": false,
  "use_cache": true,
  "session_id": "optional_session_id",
  "preferences": {
    "budget": "moderate",
    "interests": ["culture", "food", "nature"]
  }
}
```

Response:
```json
{
  "query": "Plan a 5-day trip to Japan in spring",
  "response": "Here's your personalized 5-day Japan itinerary...",
  "user_id": "user_123",
  "session_id": "session_abc123",
  "metrics": {
    "processing_time_ms": 1250,
    "ai_model_used": "gemini-1.5-flash-latest",
    "memory_context": {
      "memories_retrieved": 3,
      "context_applied": true
    },
    "cache_performance": {
      "cache_hit": false,
      "similarity_threshold": 0.85
    }
  },
  "processing_time_ms": 1250,
  "timestamp": "2025-11-29T10:30:00Z",
  "success": true,
  "ai_powered": true
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google AI API key | None | Yes |
| `AI_MODEL` | Primary AI model | `gemini-1.5-flash-latest` | No |
| `REDIS_HOST` | Redis server host | `localhost` | No |
| `REDIS_PORT` | Redis server port | `6379` | No |
| `REDIS_DB` | Redis database number | `0` | No |
| `ENVIRONMENT` | Environment name | `production` | No |

### Redis Configuration

The application supports Redis for enhanced caching performance:

```bash
# Install Redis (macOS)
brew install redis

# Start Redis server
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:alpine
```

**Note**: If Redis is unavailable, the system automatically falls back to in-memory caching.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  FastAPI App    │    │   Google AI     │
│                 │◄──►│                 │◄──►│                 │
│  Chat Interface │    │  Enterprise     │    │  Gemini Models  │
└─────────────────┘    │  Travel API     │    └─────────────────┘
                       └─────────┬───────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
        ┌───────▼───────┐ ┌─────▼─────┐ ┌───────▼───────┐
        │  Mem0 Memory  │ │   Redis   │ │   LangGraph   │
        │   Manager     │ │  Semantic │ │   Workflow    │
        │               │ │   Cache   │ │  Orchestrator │
        └───────────────┘ └───────────┘ └───────────────┘
```

### Core Components

1. **Memory Manager (Mem0)**
   - Stores and retrieves user preferences
   - Maintains conversation context
   - Enables personalized recommendations

2. **Semantic Cache (Redis)**
   - High-performance response caching
   - Similarity-based cache retrieval
   - Automatic cache invalidation

3. **Request Fingerprinter**
   - Generates unique request signatures
   - Detects duplicate requests
   - Optimizes processing efficiency

4. **Model Comparator**
   - Benchmarks AI model performance
   - Analyzes response quality
   - Tracks token usage and costs

5. **Workflow Orchestrator (LangGraph)**
   - Manages request flow
   - Handles state transitions
   - Provides error recovery

## 🔍 Usage Examples

### Basic Travel Query
```bash
curl -X POST "http://localhost:8000/memory-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Recommend romantic destinations in Europe",
    "user_id": "couple_traveler_001"
  }'
```

### Advanced Query with Preferences
```bash
curl -X POST "http://localhost:8000/memory-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Plan a budget-friendly family vacation",
    "user_id": "family_user_123",
    "include_model_comparison": true,
    "preferences": {
      "budget": "low",
      "travelers": 4,
      "ages": [35, 32, 8, 5],
      "interests": ["beaches", "theme_parks"]
    }
  }'
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

## 📈 Monitoring & Analytics

### Health Monitoring
The `/health` endpoint provides comprehensive system status:
- Service uptime and version
- Component health status
- Performance metrics
- Environment information

### Metrics Dashboard
The `/metrics` endpoint offers detailed analytics:
- Request volume and success rates
- Average response times
- Cache performance metrics
- AI model usage statistics
- Memory utilization tracking

### Performance Optimization
- **Memory**: Intelligent context retrieval reduces redundant AI calls
- **Caching**: Semantic similarity prevents duplicate processing
- **Fingerprinting**: Request deduplication optimizes throughput
- **Model Selection**: Automatic optimal model routing based on query complexity

## 🛠️ Development

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
python main.py

# Run tests
pytest tests/

# Code formatting
black .
isort .

# Type checking
mypy .
```

### Testing
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Load testing
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 📦 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations

1. **Security**
   - Use environment variables for sensitive data
   - Implement API rate limiting
   - Enable HTTPS in production
   - Regular security audits

2. **Scaling**
   - Horizontal scaling with load balancers
   - Redis clustering for cache scaling
   - Database connection pooling
   - Auto-scaling based on metrics

3. **Monitoring**
   - Application Performance Monitoring (APM)
   - Log aggregation and analysis
   - Error tracking and alerting
   - Business metrics dashboard

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Enterprise License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/enterprise-travel-assistant/issues)
- **Email**: support@enterprise-ai.com

---

**Enterprise Travel Assistant** - Powering the future of intelligent travel planning 🧳✨

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write unit tests for new features
- Update documentation as needed
- Use type hints
- Follow semantic versioning

## Performance & Monitoring

### Health Checks

- **Health endpoint**: `GET /health`
- **Metrics endpoint**: `GET /metrics`

### Logging

Logs are structured using `structlog` and can be found in:
- Development: Console output
- Production: `logs/travel_assistant.log`

### Monitoring

- Use Sentry for error tracking
- Prometheus metrics available at `/metrics`
- Health checks for uptime monitoring

## Security

- JWT-based authentication
- Input validation with Pydantic
- SQL injection protection with SQLAlchemy
- Rate limiting enabled
- HTTPS recommended for production
- Security headers implemented

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue on GitHub or contact the development team.

## Changelog

### v0.1.0
- Initial release
- Basic travel planning features
- API foundation
- Production setup