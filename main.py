"""
Travel Assistant - Main Entry Point

An intelligent travel recommendation system powered by Google Gemini AI.
Features memory management, semantic caching, and location-aware responses.

Usage:
    python main.py

Requirements:
    - Google API key (GOOGLE_API_KEY in .env)
    - Python 3.8+
    - Dependencies from requirements.txt
"""

from travel_assistant import app
import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_requirements():
    """Check if all requirements are met"""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: GOOGLE_API_KEY is required!")
        print("=" * 70)
        print("\n📋 Setup Instructions:")
        print("1. Get your API key from: https://makersuite.google.com/app/apikey")
        print("2. Open the .env file in this directory")
        print("3. Add your key: GOOGLE_API_KEY=your_actual_key_here")
        print("4. Save the file and run again")
        print("\n" + "=" * 70 + "\n")
        return False

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧳 TRAVEL ASSISTANT - PRODUCTION MODE")
    print("=" * 70)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Get configuration
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    port = int(os.getenv("PORT", 8001))

    print(f"\n✅ API Key: {api_key[:8]}...{api_key[-4:]}")
    print("✅ Gemini Flash: gemini-2.5-flash (Speed)")
    print("✅ Gemini Pro: gemini-2.5-pro (Quality)")
    print("✅ Ready for REAL AI responses")

    # Start server
    print(f"\n🚀 Starting FastAPI server on port {port}...")
    print(f"📍 Server URL: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🔧 Health Check: http://localhost:{port}/health")
    print("\n💡 Test with:")
    print(f"   curl -X POST http://localhost:{port}/memory-travel-assistant \\")
    print('        -H "Content-Type: application/json" \\')
    print(
        '        -d \'{"query": "Recommend beach destinations", "user_id": "user123"}\''
    )
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
