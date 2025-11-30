"""
Test Scenario 1.1: Single Location Queries
User asks about one location consistently
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001/memory-travel-assistant"
USER_ID = "scenario_1_1_user"


def print_response(query_num, query, response_json):
    print(f"\n{'=' * 70}")
    print(f"QUERY {query_num}: {query}")
    print(f"{'=' * 70}")
    print(f"Response Preview: {response_json['response'][:200]}...")
    print(f"\nDestinations: {response_json.get('destinations', [])}")
    print(f"Has Memory: {response_json.get('has_memory_context', False)}")
    print(f"Flash Latency: {response_json.get('flash_latency_ms', 0):.2f}ms")
    print(f"Pro Latency: {response_json.get('pro_latency_ms', 0):.2f}ms")
    print(f"Faster Model: {response_json.get('faster_model', 'N/A')}")
    print(f"{'=' * 70}\n")


def test_scenario_1_1():
    print("\n" + "=" * 70)
    print("SCENARIO 1.1: Single Location Queries")
    print("=" * 70)
    print("Testing: User asks about one location (Hyderabad) consistently")
    print("=" * 70 + "\n")

    # Query 1: restaurants in Hyderabad
    print("⏳ Sending Query 1...")
    query1 = "restaurants in Hyderabad"
    response1 = requests.post(BASE_URL, json={"query": query1, "user_id": USER_ID})
    print_response(1, query1, response1.json())

    time.sleep(2)

    # Query 2: movie theaters (should infer Hyderabad)
    print("⏳ Sending Query 2...")
    query2 = "movie theaters"
    response2 = requests.post(BASE_URL, json={"query": query2, "user_id": USER_ID})
    result2 = response2.json()
    print_response(2, query2, result2)

    # Check if it inferred Hyderabad
    if "hyderabad" in result2["response"].lower():
        print("✅ SUCCESS: Correctly inferred Hyderabad from context!")
    else:
        print("⚠️  WARNING: Did not explicitly mention Hyderabad")

    time.sleep(2)

    # Query 3: hotels (should continue with Hyderabad context)
    print("⏳ Sending Query 3...")
    query3 = "hotels"
    response3 = requests.post(BASE_URL, json={"query": query3, "user_id": USER_ID})
    result3 = response3.json()
    print_response(3, query3, result3)

    # Check if it continues with Hyderabad
    if "hyderabad" in result3["response"].lower():
        print("✅ SUCCESS: Correctly continued with Hyderabad context!")
    else:
        print("⚠️  WARNING: Did not explicitly mention Hyderabad")

    # Final summary
    print("\n" + "=" * 70)
    print("SCENARIO 1.1 TEST SUMMARY")
    print("=" * 70)
    print(f"Query 1: restaurants in Hyderabad - ✅ COMPLETED")
    print(
        f"Query 2: movie theaters - {'✅ PASSED' if 'hyderabad' in result2['response'].lower() else '⚠️ CHECK NEEDED'}"
    )
    print(
        f"Query 3: hotels - {'✅ PASSED' if 'hyderabad' in result3['response'].lower() else '⚠️ CHECK NEEDED'}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        # Check if server is running
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            test_scenario_1_1()
        else:
            print("❌ Server is not responding properly")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Server is not running!")
        print("Please start the server first with: python3 main.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
