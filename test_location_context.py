#!/usr/bin/env python3
"""
Test script to demonstrate location context handling
This tests the scenario:
1. Ask about restaurants in Hyderabad
2. Ask about movie theaters (should be Hyderabad)
3. Ask about restaurants in Goa (should switch to Goa)
4. Ask about movie theaters (should still be Goa, NOT Hyderabad)
"""

import requests
import json
import time

API_URL = "http://localhost:8001/memory-travel-assistant"
USER_ID = "location_test_user"


def send_query(query: str, step: int):
    """Send a query and display response"""
    print(f"\n{'=' * 70}")
    print(f"STEP {step}: {query}")
    print("=" * 70)

    response = requests.post(
        API_URL,
        json={"query": query, "user_id": USER_ID},
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Response received:")
        print(f"   Length: {len(data.get('response', ''))} chars")

        # Extract key information
        response_text = data.get("response", "")
        response_lower = response_text.lower()

        # Check for location mentions
        has_hyderabad = "hyderabad" in response_lower
        has_goa = "goa" in response_lower

        print(f"\n📍 Location Analysis:")
        print(f"   Mentions Hyderabad: {'✅ YES' if has_hyderabad else '❌ NO'}")
        print(f"   Mentions Goa: {'✅ YES' if has_goa else '❌ NO'}")

        # Show preview
        preview = (
            response_text[:300] + "..." if len(response_text) > 300 else response_text
        )
        print(f"\n📝 Response Preview:")
        print(f"   {preview}")

        # Show workflow logs if available
        if "workflow_logs" in data:
            logs = data["workflow_logs"]
            if "memory_retrieved" in logs:
                print(f"\n💾 Memory Context:")
                print(f"   {logs['memory_retrieved'][:150]}...")

        return data
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"   {response.text}")
        return None


def main():
    """Run the location context test"""
    print("\n" + "=" * 70)
    print("🧪 LOCATION CONTEXT TEST")
    print("=" * 70)
    print("\nThis test demonstrates proper location context handling:")
    print("1. Query about Hyderabad restaurants")
    print("2. Query about movie theaters (should stay in Hyderabad)")
    print("3. Query about Goa restaurants (switch location)")
    print("4. Query about movie theaters (should now be about Goa)")
    print("\n" + "=" * 70)

    # Test sequence
    queries = [
        "Recommend best restaurants in Hyderabad",
        "What about movie theaters?",
        "Tell me about restaurants in Goa",
        "What about movie theaters?",
    ]

    for i, query in enumerate(queries, 1):
        send_query(query, i)
        time.sleep(2)  # Small delay between requests

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print("\nExpected Behavior:")
    print("  Step 1: Should mention Hyderabad restaurants")
    print("  Step 2: Should mention Hyderabad movie theaters")
    print("  Step 3: Should mention Goa restaurants (NOT Hyderabad)")
    print("  Step 4: Should mention Goa movie theaters (NOT Hyderabad)")
    print("\nIf Step 4 mentions Hyderabad, there's a location context bug.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
