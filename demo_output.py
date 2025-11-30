#!/usr/bin/env python3
"""
Demo script to show expected output format
This runs the travel assistant and shows all workflow steps
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from travel_assistant import (
    MemoryManager,
    SemanticCache,
    RequestFingerprinter,
    GeminiModelComparator,
    TravelAssistantWorkflow,
)


def main():
    print("\n" + "=" * 70)
    print("🧳 TRAVEL ASSISTANT DEMO - Expected Output Format")
    print("=" * 70)

    # Initialize components
    memory = MemoryManager()
    cache = SemanticCache()
    fingerprinter = RequestFingerprinter()
    comparator = GeminiModelComparator()

    workflow = TravelAssistantWorkflow(memory, cache, fingerprinter, comparator)

    # Test query with preferences
    user_id = "demo_user"
    query = (
        "I prefer quiet locations and vegetarian food. Recommend beach destinations."
    )

    print(f"\n📝 Query: {query}")
    print(f"👤 User ID: {user_id}\n")

    # Store preferences first
    memory.store_preference(user_id, "prefers quiet locations")
    memory.store_preference(user_id, "prefers vegetarian food")

    # Process query
    print("🔄 Processing through LangGraph workflow...\n")
    result = workflow.process_query(query, user_id)

    print("\n" + "=" * 70)
    print("✅ FINAL RESULT:")
    print("=" * 70)
    print(f"Response Preview: {result['response'][:200]}...")
    print(f"\nMetadata: {result['metadata']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
