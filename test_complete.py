#!/usr/bin/env python3
"""Test the API to verify output format"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"


def test_api():
    print("🧪 Testing Travel Assistant API\n")

    # Test 1: First query with preferences
    print("=" * 70)
    print("TEST 1: Query with preferences")
    print("=" * 70)

    response1 = requests.post(
        f"{BASE_URL}/memory-travel-assistant",
        json={
            "query": "I prefer quiet locations and vegetarian food. Recommend beach destinations.",
            "user_id": "demo_user",
        },
        timeout=120,
    )

    print(f"\n✅ Status Code: {response1.status_code}")
    print(f"\n📄 Response JSON:")
    print(json.dumps(response1.json(), indent=2))

    # Wait a bit
    time.sleep(2)

    # Test 2: Second query (should use memory)
    print("\n\n" + "=" * 70)
    print("TEST 2: Follow-up query (should use memory)")
    print("=" * 70)

    response2 = requests.post(
        f"{BASE_URL}/memory-travel-assistant",
        json={"query": "hyderabad", "user_id": "demo_user"},
        timeout=120,
    )

    print(f"\n✅ Status Code: {response2.status_code}")
    print(f"\n📄 Response JSON:")
    result = response2.json()
    print(json.dumps(result, indent=2))

    print("\n\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    print(f"Response has 'query' field: {'query' in result}")
    print(f"Response has 'response' field: {'response' in result}")
    print(f"Response has 'metadata' field: {'metadata' in result}")
    print(f"Response has 'timestamp' field: {'timestamp' in result}")
    print(
        f"Has memory context: {result.get('metadata', {}).get('has_memory_context', False)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Server not running on port 8001")
        print("Start with: python3 main.py")
    except Exception as e:
        print(f"❌ Error: {e}")
