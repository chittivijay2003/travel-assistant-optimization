#!/usr/bin/env python3
"""
Streamlit App Runner
Launch the Enterprise Travel Assistant Dashboard
"""

import subprocess
import sys
import os
from pathlib import Path


def run_streamlit_app():
    """Run the Streamlit dashboard application"""

    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    # Path to Streamlit app
    app_path = project_root / "travel_assistant" / "ui" / "streamlit_app.py"

    if not app_path.exists():
        print(f"❌ Streamlit app not found at: {app_path}")
        return

    print("🚀 Starting Enterprise Travel Assistant Dashboard...")
    print(f"📁 App location: {app_path}")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("📡 Make sure FastAPI server is running at: http://localhost:8000")
    print("-" * 60)

    try:
        # Run Streamlit app
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                "8501",
                "--server.address",
                "0.0.0.0",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    run_streamlit_app()
