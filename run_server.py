#!/usr/bin/env python3
"""
FastAPI Server Runner
Launch the Enterprise Travel Assistant API Server
"""

import subprocess
import sys
from pathlib import Path


def run_fastapi_server():
    """Run the FastAPI server"""

    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    print("🚀 Starting Enterprise Travel Assistant API Server...")
    print("📁 Project root:", project_root)
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print("💬 Chat interface at: http://localhost:8000/chat")
    print("-" * 60)

    try:
        # Run FastAPI server with uvicorn
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
                "--log-level",
                "info",
            ],
            check=True,
            cwd=project_root,
        )

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running FastAPI server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    run_fastapi_server()
