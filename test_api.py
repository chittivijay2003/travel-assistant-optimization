#!/usr/bin/env python3
"""Test the API endpoint"""

import requests
import json

url = "http://localhost:8001/memory-travel-assistant"
data = {"query": "Recommend beach destinations in Thailand", "user_id": "test123"}

try:
    print("🧪 Testing Travel Assistant API...")
    print(f"📡 Sending request to: {url}")
    print(f"📝 Query: {data['query']}\n")

    response = requests.post(url, json=data, timeout=30)

    print(f"✅ Status Code: {response.status_code}\n")

    if response.status_code == 200:
        result = response.json()
        print("=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 70)

        # Check if it's a real AI response
        if result["metadata"]["latency_ms"] > 0:
            print("✅ Real AI response received!")
            print(f"⚡ Latency: {result['metadata']['latency_ms']}ms")
            print(f"🤖 Model: {result['metadata']['model']}")
        else:
            print("❌ Mock/fallback response")
    else:
        print(f"❌ Error: {response.text}")

except requests.exceptions.ConnectionRefusedError:
    print("❌ Server not running on port 8001")
except Exception as e:
    print(f"❌ Error: {e}")
