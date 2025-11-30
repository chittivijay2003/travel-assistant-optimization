"""
🧳 Travel Assistant - Complete Implementation
GenAI Day 5 Assignment

This module implements a complete AI-powered travel assistant with:
- Mem0 memory management
- Redis semantic caching
- Request fingerprinting
- Gemini Flash vs Pro comparison
- LangGraph workflow integration
- FastAPI REST endpoint
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Google Gemini AI
import google.generativeai as genai

# Memory management
from mem0 import Memory

# Caching
import redis

# Semantic similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Web framework
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", 0.85))

# Configure Gemini - REQUIRED for production
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini API configured successfully")
else:
    print("⚠️  Warning: No GOOGLE_API_KEY - Some features will be limited")


# ============================================================================
# TASK 2: Mem0 Memory Management
# ============================================================================


class MemoryManager:
    """
    Manages user preferences and conversation history using Mem0.
    Provides fallback storage when Mem0 is unavailable.
    """

    def __init__(self):
        """Initialize Mem0 with fallback support"""
        try:
            self.memory = Memory()
            self.fallback_storage = {}
            self.use_fallback = False
            print("✅ Mem0 initialized")
        except Exception as e:
            print(f"⚠️  Mem0 unavailable, using fallback: {str(e)[:50]}")
            self.memory = None
            self.fallback_storage = {}
            self.use_fallback = True

    def store_preference(self, user_id: str, preference: str) -> bool:
        """Store user preference"""
        try:
            if not self.use_fallback and self.memory:
                self.memory.add(preference, user_id=user_id)
            else:
                if user_id not in self.fallback_storage:
                    self.fallback_storage[user_id] = []
                self.fallback_storage[user_id].append(
                    {"content": preference, "timestamp": datetime.now().isoformat()}
                )
            return True
        except Exception as e:
            print(f"❌ Store failed: {e}")
            return False

    def retrieve_context(self, user_id: str, query: str, limit: int = 3) -> List[str]:
        """Retrieve relevant memories"""
        try:
            if not self.use_fallback and self.memory:
                results = self.memory.search(query, user_id=user_id, limit=limit)
                return [r.get("memory", "") for r in results if r.get("memory")]
            else:
                memories = self.fallback_storage.get(user_id, [])
                return [m["content"] for m in memories][:limit]
        except Exception as e:
            print(f"❌ Retrieve failed: {e}")
            return []

    def update_memory(self, user_id: str, conversation: str) -> bool:
        """Update memory with conversation"""
        return self.store_preference(user_id, f"Conversation: {conversation}")


# ============================================================================
# TASK 3: Redis Semantic Cache
# ============================================================================


class SemanticCache:
    """
    Semantic caching using Redis and sentence embeddings.
    Caches responses and retrieves similar queries using cosine similarity.
    """

    def __init__(self, threshold: float = CACHE_THRESHOLD, ttl: int = CACHE_TTL):
        """Initialize semantic cache"""
        self.threshold = threshold
        self.ttl = ttl
        self.fallback_cache = {}

        # Initialize Redis
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
            )
            self.redis_client.ping()
            self.use_redis = True
            print(f"✅ Redis connected ({REDIS_HOST}:{REDIS_PORT})")
        except Exception as e:
            print(f"⚠️  Redis unavailable, using fallback: {str(e)[:50]}")
            self.redis_client = None
            self.use_redis = False

        # Initialize sentence transformer
        try:
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Sentence encoder loaded")
        except Exception as e:
            print(f"⚠️  Encoder unavailable: {e}")
            self.encoder = None

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.encoder:
            return self.encoder.encode([text])[0]
        return np.zeros(384)  # Fallback zero vector

    def cache_response(self, query: str, response: str, model: str) -> None:
        """Cache a response"""
        try:
            embedding = self._generate_embedding(query)
            cache_data = {
                "query": query,
                "response": response,
                "model": model,
                "embedding": embedding.tolist(),
                "timestamp": datetime.now().isoformat(),
            }

            cache_key = f"cache:{model}:{hashlib.md5(query.encode()).hexdigest()}"

            if self.use_redis and self.redis_client:
                self.redis_client.setex(cache_key, self.ttl, json.dumps(cache_data))
            else:
                self.fallback_cache[cache_key] = cache_data

        except Exception as e:
            print(f"❌ Cache store failed: {e}")

    def get_cached_response(self, query: str, model: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if similar query exists"""
        try:
            query_embedding = self._generate_embedding(query)

            # Get all cached entries for this model
            if self.use_redis and self.redis_client:
                pattern = f"cache:{model}:*"
                keys = self.redis_client.keys(pattern)
                cache_entries = []
                for key in keys:
                    data = self.redis_client.get(key)
                    if data:
                        cache_entries.append(json.loads(data))
            else:
                cache_entries = [
                    v
                    for k, v in self.fallback_cache.items()
                    if k.startswith(f"cache:{model}:")
                ]

            # Find best semantic match
            best_similarity = 0.0
            best_match = None

            for entry in cache_entries:
                cached_embedding = np.array(entry["embedding"])
                similarity = cosine_similarity([query_embedding], [cached_embedding])[
                    0
                ][0]

                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry

            if best_match:
                return {
                    "response": best_match["response"],
                    "similarity": float(best_similarity),
                    "cached_query": best_match["query"],
                }

            return None

        except Exception as e:
            print(f"❌ Cache retrieval failed: {e}")
            return None


# ============================================================================
# TASK 4: Request Fingerprinting
# ============================================================================


class RequestFingerprinter:
    """
    Generates unique fingerprints for requests to detect duplicates.
    Uses SHA-256 hashing of normalized query content.
    """

    def __init__(self):
        """Initialize fingerprinter"""
        self.fingerprint_history = {}

    def generate_fingerprint(self, query: str, user_id: str) -> Dict[str, Any]:
        """Generate request fingerprint"""
        # Normalize query
        normalized_query = query.lower().strip()

        # Create fingerprint data
        fingerprint_data = f"{user_id}:{normalized_query}:{datetime.now().date()}"
        fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()

        # Check if duplicate
        is_duplicate = fingerprint_hash in self.fingerprint_history

        if is_duplicate:
            self.fingerprint_history[fingerprint_hash]["count"] += 1
        else:
            self.fingerprint_history[fingerprint_hash] = {
                "query": query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "count": 1,
            }

        return {
            "fingerprint": fingerprint_hash,
            "is_duplicate": is_duplicate,
            "count": self.fingerprint_history[fingerprint_hash]["count"],
            "first_seen": self.fingerprint_history[fingerprint_hash]["timestamp"],
        }


# ============================================================================
# TASK 5: Model Comparison (Gemini Flash vs Pro)
# ============================================================================


class GeminiModelComparator:
    """
    Compares Gemini Flash and Pro models on:
    - Response quality
    - Response length
    - Latency
    - Token usage
    """

    def __init__(self):
        """Initialize both models"""
        if GOOGLE_API_KEY:
            # Using Gemini 2.5 models (Flash for speed, Pro for quality)
            self.flash_model = genai.GenerativeModel("gemini-2.5-flash")
            self.pro_model = genai.GenerativeModel("gemini-2.5-pro")
            self.api_available = True
            print("✅ Gemini 2.5 models initialized (Flash + Pro)")
        else:
            self.flash_model = None
            self.pro_model = None
            self.api_available = False
            print("❌ API key missing - Model comparison unavailable")

    def compare_models(self, prompt: str) -> Dict[str, Any]:
        """Compare both models on the same prompt"""
        results = {"prompt": prompt, "flash": {}, "pro": {}, "comparison": {}}

        if not self.api_available:
            raise ValueError(
                "API key not configured. Add GOOGLE_API_KEY to .env file to use real Gemini models."
            )

        # Flash model
        start_time = time.time()
        try:
            flash_response = self.flash_model.generate_content(prompt)
            flash_text = flash_response.text
            flash_latency = (time.time() - start_time) * 1000
        except Exception as e:
            flash_text = f"Error: {e}"
            flash_latency = 0

        results["flash"] = {
            "response": flash_text,
            "latency_ms": round(flash_latency, 2),
            "length": len(flash_text),
            "word_count": len(flash_text.split()),
        }

        # Pro model
        start_time = time.time()
        try:
            pro_response = self.pro_model.generate_content(prompt)
            pro_text = pro_response.text
            pro_latency = (time.time() - start_time) * 1000
        except Exception as e:
            pro_text = f"Error: {e}"
            pro_latency = 0

        results["pro"] = {
            "response": pro_text,
            "latency_ms": round(pro_latency, 2),
            "length": len(pro_text),
            "word_count": len(pro_text.split()),
        }

        # Comparison metrics
        results["comparison"] = {
            "faster_model": "flash"
            if results["flash"]["latency_ms"] < results["pro"]["latency_ms"]
            else "pro",
            "more_detailed": "pro"
            if results["pro"]["length"] > results["flash"]["length"]
            else "flash",
            "speed_difference_ms": abs(
                results["flash"]["latency_ms"] - results["pro"]["latency_ms"]
            ),
            "length_difference": abs(
                results["flash"]["length"] - results["pro"]["length"]
            ),
        }

        return results


# ============================================================================
# TASK 6: LangGraph Travel Assistant Workflow
# ============================================================================


class TravelAssistantState(TypedDict):
    """State definition for LangGraph workflow"""

    query: str
    user_id: str
    fingerprint: Dict[str, Any]
    memory_context: List[str]
    cached_response: Optional[Dict[str, Any]]
    model_comparison: Optional[Dict[str, Any]]
    final_response: str
    metadata: Dict[str, Any]


class TravelAssistantWorkflow:
    """
    LangGraph workflow integrating:
    - Memory retrieval
    - Semantic caching
    - Request fingerprinting
    - AI response generation
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        cache: SemanticCache,
        fingerprinter: RequestFingerprinter,
        model_comparator: GeminiModelComparator,
    ):
        """Initialize workflow with all components"""
        self.memory = memory_manager
        self.cache = cache
        self.fingerprinter = fingerprinter
        self.comparator = model_comparator
        self.workflow = self._build_workflow()
        print("✅ LangGraph workflow built")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(TravelAssistantState)

        # Add nodes
        workflow.add_node("fingerprint_request", self._fingerprint_node)
        workflow.add_node("check_cache", self._cache_check_node)
        workflow.add_node("retrieve_memory", self._memory_retrieval_node)
        workflow.add_node("generate_response", self._generation_node)
        workflow.add_node("update_memory", self._memory_update_node)

        # Define edges
        workflow.set_entry_point("fingerprint_request")
        workflow.add_edge("fingerprint_request", "check_cache")

        # Conditional routing from cache check
        workflow.add_conditional_edges(
            "check_cache",
            self._route_after_cache,
            {"use_cache": END, "generate_new": "retrieve_memory"},
        )

        workflow.add_edge("retrieve_memory", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def _fingerprint_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """Generate request fingerprint"""
        state["fingerprint"] = self.fingerprinter.generate_fingerprint(
            state["query"], state["user_id"]
        )
        return state

    def _cache_check_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """Check if response is cached"""
        cached = self.cache.get_cached_response(state["query"], "gemini-flash")

        if cached:
            state["cached_response"] = cached
            state["final_response"] = cached["response"]
            state["metadata"] = {
                "source": "cache",
                "similarity": cached["similarity"],
                "cached_query": cached["cached_query"],
            }
        else:
            state["cached_response"] = None

        return state

    def _memory_retrieval_node(
        self, state: TravelAssistantState
    ) -> TravelAssistantState:
        """Retrieve user memory context"""
        state["memory_context"] = self.memory.retrieve_context(
            state["user_id"], state["query"]
        )

        # Print memory context
        if state["memory_context"]:
            print("\n" + "=" * 70)
            print("📝 Memory Retrieved:")
            for ctx in state["memory_context"]:
                print(f"   {ctx}")
            print("=" * 70)

        return state

    def _generation_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """Generate AI response"""
        # Build prompt with memory context
        prompt = state["query"]
        if state["memory_context"]:
            context_str = "\n".join(state["memory_context"])
            prompt = f"""User Preferences and History (for reference):
{context_str}

Current Query: {state["query"]}

IMPORTANT INSTRUCTIONS:
1. Focus on the CURRENT query and any location/topic mentioned in it
2. If the current query mentions a specific location (e.g., "in Goa"), provide information ONLY about that location
3. Use the user preferences/history ONLY for understanding their general interests, NOT for location context
4. If the current query asks about restaurants, hotels, or activities in a specific place, answer about THAT place only
5. Do NOT mix information from previous locations with the current query's location

Please provide a personalized response based on these instructions."""

        # Generate response using model comparison
        comparison = self.comparator.compare_models(prompt)
        state["model_comparison"] = comparison

        # Print model comparison
        print("\n" + "=" * 70)
        print("🤖 GEMINI FLASH RESPONSE:")
        print(f"   Latency: {comparison['flash']['latency_ms']:.2f}ms")
        print(f"   Length: {comparison['flash']['length']} chars")
        print(f"   Preview: {comparison['flash']['response'][:150]}...")
        print("\n" + "=" * 70)
        print("🤖 GEMINI PRO RESPONSE:")
        print(f"   Latency: {comparison['pro']['latency_ms']:.2f}ms")
        print(f"   Length: {comparison['pro']['length']} chars")
        print(f"   Preview: {comparison['pro']['response'][:150]}...")
        print(f"\n⚡ Faster Model: {comparison['comparison']['faster_model']}")
        print("=" * 70)

        # Use Flash model response (faster)
        state["final_response"] = comparison["flash"]["response"]
        state["metadata"] = {
            "source": "ai_generated",
            "model": "gemini-flash",
            "latency_ms": comparison["flash"]["latency_ms"],
            "has_memory_context": len(state["memory_context"]) > 0,
        }

        # Cache the response
        self.cache.cache_response(
            state["query"], state["final_response"], "gemini-flash"
        )

        return state

    def _memory_update_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """Update user memory with conversation"""
        # Extract location from query if present (simple pattern matching)
        location = None

        # Common patterns: "in [location]", "at [location]", "[location]" followed by context words
        import re

        location_patterns = [
            r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "in Hyderabad", "in New Delhi"
            r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",  # "at Goa"
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:restaurants|hotels|movie|theaters|places)",  # "Goa restaurants"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, state["query"])
            if match:
                location = match.group(1)
                break

        # Store with location context if found
        if location:
            conversation = f"[Location: {location}] Query: {state['query']}\nResponse: {state['final_response'][:200]}"
        else:
            conversation = (
                f"Query: {state['query']}\nResponse: {state['final_response'][:200]}"
            )

        self.memory.update_memory(state["user_id"], conversation)
        print("\n✅ Memory Updated.")
        return state

    def _route_after_cache(self, state: TravelAssistantState) -> str:
        """Route based on cache hit/miss"""
        if state.get("cached_response"):
            return "use_cache"
        return "generate_new"

    def process_query(
        self, query: str, user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """Process a travel query through the workflow"""
        initial_state: TravelAssistantState = {
            "query": query,
            "user_id": user_id,
            "fingerprint": {},
            "memory_context": [],
            "cached_response": None,
            "model_comparison": None,
            "final_response": "",
            "metadata": {},
        }

        final_state = self.workflow.invoke(initial_state)

        # Build workflow logs for response
        workflow_logs = {}

        # Memory Retrieved
        if final_state.get("memory_context"):
            workflow_logs["memory_retrieved"] = ", ".join(final_state["memory_context"])

        # Model Comparison (if available)
        if final_state.get("model_comparison"):
            comp = final_state["model_comparison"]
            workflow_logs["gemini_flash_response"] = {
                "latency_ms": comp["flash"]["latency_ms"],
                "length": comp["flash"]["length"],
                "preview": comp["flash"]["response"][:100] + "...",
            }
            workflow_logs["gemini_pro_response"] = {
                "latency_ms": comp["pro"]["latency_ms"],
                "length": comp["pro"]["length"],
                "preview": comp["pro"]["response"][:100] + "...",
            }
            workflow_logs["faster_model"] = comp["comparison"]["faster_model"]

        # Extract recommended destinations
        response_text = final_state["final_response"]
        import re

        destinations = re.findall(
            r"\*\*([A-Z][a-zA-Z\s]+(?:\([^)]+\))?)[:\*]", response_text[:800]
        )
        if destinations:
            workflow_logs["recommended_destinations"] = [
                dest.strip() for dest in destinations[:5]
            ]

        # Memory Updated
        workflow_logs["memory_updated"] = True

        # Cached Fingerprint
        if final_state.get("fingerprint"):
            workflow_logs["cached_fingerprint"] = (
                final_state["fingerprint"].get("hash", "N/A")[:16] + "..."
            )

        # Print to console
        if final_state.get("fingerprint"):
            print(
                f"\n🔐 Cached Fingerprint: {final_state['fingerprint'].get('hash', 'N/A')[:16]}..."
            )

        if workflow_logs.get("recommended_destinations"):
            print("\n" + "=" * 70)
            print("✈️  RECOMMENDED DESTINATIONS:")
            for dest in workflow_logs["recommended_destinations"]:
                print(f"   - {dest}")
            print("=" * 70 + "\n")

        return {
            "query": query,
            "response": final_state["final_response"],
            "user_id": user_id,
            "metadata": final_state["metadata"],
            "workflow_logs": workflow_logs,
            "fingerprint": final_state["fingerprint"],
        }


# ============================================================================
# TASK 7: FastAPI Endpoint
# ============================================================================


class TravelQueryRequest(BaseModel):
    """Request model for travel assistant"""

    query: str
    user_id: str = "anonymous"
    include_model_comparison: bool = False


class TravelQueryResponse(BaseModel):
    """Response model for travel assistant"""

    query: str
    response: str
    user_id: str
    metadata: Dict[str, Any]
    workflow_logs: Dict[str, Any]
    timestamp: str


# Initialize all components
print("\n" + "=" * 70)
print("🚀 Initializing Travel Assistant Components")
print("=" * 70)

memory_manager = MemoryManager()
semantic_cache = SemanticCache()
fingerprinter = RequestFingerprinter()
model_comparator = GeminiModelComparator()
travel_assistant = TravelAssistantWorkflow(
    memory_manager, semantic_cache, fingerprinter, model_comparator
)

print("=" * 70)
print("✅ All components initialized successfully!")
print("=" * 70 + "\n")

# Create FastAPI app
app = FastAPI(
    title="Travel Assistant API",
    description="AI-powered travel assistant with memory, caching, and intelligent routing",
    version="1.0.0",
)


@app.post("/memory-travel-assistant", response_model=TravelQueryResponse)
async def memory_travel_assistant_endpoint(request: TravelQueryRequest):
    """
    Main travel assistant endpoint

    Features:
    - Memory-aware responses
    - Semantic caching
    - Request fingerprinting
    - Optional model comparison
    """
    try:
        # Process query through workflow
        result = travel_assistant.process_query(request.query, request.user_id)

        # Add model comparison if requested
        if (
            request.include_model_comparison
            and result["metadata"].get("source") != "cache"
        ):
            comparison = model_comparator.compare_models(request.query)
            result["metadata"]["model_comparison"] = {
                "flash_latency_ms": comparison["flash"]["latency_ms"],
                "pro_latency_ms": comparison["pro"]["latency_ms"],
                "faster_model": comparison["comparison"]["faster_model"],
                "flash_response_length": comparison["flash"]["length"],
                "pro_response_length": comparison["pro"]["length"],
            }

        return TravelQueryResponse(
            query=request.query,
            response=result["response"],
            user_id=request.user_id,
            metadata=result["metadata"],
            workflow_logs=result.get("workflow_logs", {}),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Travel Assistant API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "main": "/memory-travel-assistant",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "memory": "operational",
            "cache": "operational",
            "models": "operational" if GOOGLE_API_KEY else "demo_mode",
        },
    }


# ============================================================================
# DEMONSTRATION
# ============================================================================


def run_demonstration():
    """Run complete demonstration of all features"""
    print("\n" + "=" * 70)
    print("🧳 TRAVEL ASSISTANT DEMONSTRATION")
    print("=" * 70)

    demo_user = "demo_user_001"
    demo_query = "Plan a beach vacation. I prefer quiet locations and vegetarian food."

    print(f"\n👤 User ID: {demo_user}")
    print(f"📝 Query: {demo_query}")

    # Step 1: Store user preferences
    print("\n" + "-" * 70)
    print("STEP 1: Storing User Preferences")
    print("-" * 70)
    memory_manager.store_preference(demo_user, "Prefers quiet, secluded beaches")
    memory_manager.store_preference(demo_user, "Strict vegetarian diet")
    memory_manager.store_preference(
        demo_user, "Enjoys cultural activities and local museums"
    )

    # Step 2: First query (cache miss)
    print("\n" + "-" * 70)
    print("STEP 2: First Query (Expected: AI Generation)")
    print("-" * 70)
    result1 = travel_assistant.process_query(demo_query, demo_user)
    print(f"\n✅ Response Source: {result1['metadata']['source']}")
    print(f"🔑 Fingerprint: {result1['fingerprint']['fingerprint'][:20]}...")
    print(f"📊 Is Duplicate: {result1['fingerprint']['is_duplicate']}")
    print(f"\n💬 Response:\n{result1['response'][:200]}...")

    # Step 3: Same query again (cache hit)
    print("\n" + "-" * 70)
    print("STEP 3: Same Query Again (Expected: Cache Hit)")
    print("-" * 70)
    result2 = travel_assistant.process_query(demo_query, demo_user)
    print(f"\n✅ Response Source: {result2['metadata']['source']}")
    if "similarity" in result2["metadata"]:
        print(f"🎯 Cache Similarity: {result2['metadata']['similarity']:.4f}")
    print(f"📊 Is Duplicate: {result2['fingerprint']['is_duplicate']}")

    # Step 4: Model comparison
    print("\n" + "-" * 70)
    print("STEP 4: Model Comparison (Flash vs Pro)")
    print("-" * 70)
    comparison = model_comparator.compare_models("Quick beach recommendation")
    print(
        f"\n⚡ Flash: {comparison['flash']['latency_ms']}ms, {comparison['flash']['length']} chars"
    )
    print(
        f"🎯 Pro: {comparison['pro']['latency_ms']}ms, {comparison['pro']['length']} chars"
    )
    print(f"🏆 Faster: {comparison['comparison']['faster_model'].upper()}")
    print(f"📄 More Detailed: {comparison['comparison']['more_detailed'].upper()}")

    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE - ALL TASKS FUNCTIONAL!")
    print("=" * 70 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run demonstration
    run_demonstration()

    # Start FastAPI server
    print("\n🚀 Starting FastAPI server...")
    print("📍 Server will be available at: http://localhost:8001")
    print("📚 API documentation at: http://localhost:8001/docs")
    print("\n💡 Test with curl:")
    print("   curl -X POST http://localhost:8001/memory-travel-assistant \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"query": "Plan beach vacation", "user_id": "user123"}\'')
    print("\n" + "=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
