# Travel Assistant Optimization

An AI-powered travel assistant application designed to optimize travel planning and provide intelligent recommendations for travelers.

## Features

- 🗺️ Intelligent trip planning and optimization
- 🌤️ Real-time weather integration
- 🏨 Hotel and accommodation recommendations
- ✈️ Flight search and booking assistance
- 🚗 Transportation optimization
- 📊 Travel analytics and insights
- 🔐 Secure user authentication
- 📱 RESTful API with FastAPI
- 🚀 Production-ready deployment

## Quick Start

### Prerequisites

- Python 3.13+
- UV package manager
- Redis (for caching)
- PostgreSQL (optional, for production)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/travel-assistant-optimization.git
   cd travel-assistant-optimization
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration values
   ```

4. **Run the application**
   ```bash
   uv run python main.py
   ```

## Environment Configuration

### Development Setup

For development, copy `.env.dev` to `.env`:

```bash
cp .env.dev .env
```

### Production Setup

For production, copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with production values
```

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `APP_NAME` | Application name | ✅ |
| `ENVIRONMENT` | Environment (development/production) | ✅ |
| `SECRET_KEY` | Application secret key | ✅ |
| `OPENWEATHER_API_KEY` | OpenWeather API key | ✅ |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key | ✅ |
| `DATABASE_URL` | Database connection URL | ❌ |
| `REDIS_URL` | Redis connection URL | ❌ |

## Development

### Install Development Dependencies

```bash
uv sync --group dev
```

### Code Quality

Run code formatting and linting:

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy .

# Security check
uv run bandit -r .
```

### Testing

Run tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=travel_assistant

# Run specific test types
uv run pytest -m unit
uv run pytest -m integration
```

### Pre-commit Hooks

Set up pre-commit hooks:

```bash
uv run pre-commit install
```

## Production Deployment

### Using Docker (Recommended)

```bash
# Build the image
docker build -t travel-assistant .

# Run the container
docker run -d \
  --name travel-assistant \
  -p 8000:8000 \
  --env-file .env \
  travel-assistant
```

### Using Gunicorn

```bash
# Install production dependencies
uv sync --group prod

# Run with Gunicorn
uv run gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables for Production

Ensure these are set for production:

- `ENVIRONMENT=production`
- `DEBUG=false`
- `SECRET_KEY` (strong, unique key)
- `DATABASE_URL` (PostgreSQL recommended)
- `REDIS_URL` (for caching)
- `SENTRY_DSN` (for error tracking)

## API Documentation

Once the application is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
travel-assistant-optimization/
├── travel_assistant/           # Main application package
│   ├── __init__.py
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration settings
│   ├── models/                # Data models
│   ├── api/                   # API routes
│   ├── services/              # Business logic
│   ├── utils/                 # Utility functions
│   └── dependencies.py        # FastAPI dependencies
├── tests/                     # Test files
├── migrations/                # Database migrations
├── logs/                      # Log files
├── .env.example              # Environment variables template
├── .env.dev                  # Development environment
├── .gitignore                # Git ignore rules
├── pyproject.toml            # Project configuration
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # This file
```

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