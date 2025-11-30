"""
Quick test script to verify Travel Assistant is working
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("\n" + "=" * 70)
print("🧪 TRAVEL ASSISTANT - CONFIGURATION CHECK")
print("=" * 70)

# Check API Key
api_key = os.getenv("GOOGLE_API_KEY", "").strip()
if api_key:
    print(f"\n✅ GOOGLE_API_KEY: {api_key[:8]}...{api_key[-4:]}")
    print("   Status: READY FOR PRODUCTION")
else:
    print("\n❌ GOOGLE_API_KEY: Not configured")
    print("   Status: REQUIRES SETUP")
    print("\n📋 To fix:")
    print("   1. Open .env file")
    print("   2. Add: GOOGLE_API_KEY=your_key_here")
    print("   3. Save and run again")

# Check other config
print(f"\n📊 Configuration:")
print(f"   Redis Host: {os.getenv('REDIS_HOST', 'localhost')}")
print(f"   Redis Port: {os.getenv('REDIS_PORT', '6379')}")
print(f"   Cache TTL: {os.getenv('CACHE_TTL', '3600')}s")
print(f"   Cache Threshold: {os.getenv('CACHE_THRESHOLD', '0.85')}")
print(f"   Server Port: {os.getenv('PORT', '8001')}")

# Test imports
print(f"\n📦 Testing imports...")
try:
    import google.generativeai as genai

    print("   ✅ google-generativeai")
except:
    print("   ❌ google-generativeai (run: pip install google-generativeai)")

try:
    from mem0 import Memory

    print("   ✅ mem0ai")
except:
    print("   ❌ mem0ai (run: pip install mem0ai)")

try:
    import redis

    print("   ✅ redis")
except:
    print("   ❌ redis (run: pip install redis)")

try:
    from sentence_transformers import SentenceTransformer

    print("   ✅ sentence-transformers")
except:
    print("   ❌ sentence-transformers (run: pip install sentence-transformers)")

try:
    from langgraph.graph import StateGraph

    print("   ✅ langgraph")
except:
    print("   ❌ langgraph (run: pip install langgraph)")

try:
    from fastapi import FastAPI

    print("   ✅ fastapi")
except:
    print("   ❌ fastapi (run: pip install fastapi)")

print("\n" + "=" * 70)

if api_key:
    print("\n🚀 Ready to run! Execute: python3 main.py")
else:
    print("\n⚠️  Configure .env first, then run: python3 main.py")

print("=" * 70 + "\n")
