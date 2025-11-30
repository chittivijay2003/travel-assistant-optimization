#!/usr/bin/env python3
"""Test to show workflow_logs in response"""

import requests
import json

print("Starting server test...\n")

try:
    response = requests.post(
        "http://localhost:8001/memory-travel-assistant",
        json={
            "query": "I love quiet beaches with excellent vegetarian cuisine. Suggest destinations.",
            "user_id": "new_test_user",
        },
        timeout=120,
    )

    result = response.json()

    print("=" * 70)
    print("JSON RESPONSE WITH WORKFLOW_LOGS:")
    print("=" * 70)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 70)
    print("WORKFLOW_LOGS SECTION:")
    print("=" * 70)
    print(json.dumps(result.get("workflow_logs", {}), indent=2))

except Exception as e:
    print(f"Error: {e}")
