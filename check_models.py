#!/usr/bin/env python3
"""Check available Gemini models"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ No API key found")
    exit(1)

genai.configure(api_key=api_key)

print("✅ API Key configured")
print("\n" + "=" * 70)
print("Available Gemini Models:")
print("=" * 70)

try:
    models = genai.list_models()
    for model in models:
        if "generateContent" in model.supported_generation_methods:
            print(f"\n📦 {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Supported Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"❌ Error listing models: {e}")
