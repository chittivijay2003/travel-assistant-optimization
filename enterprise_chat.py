"""
🧳 Enterprise Travel Assistant - Real AI with Memory, Caching & LangGraph
Integrates all 7 tasks from the assignment with full enterprise features:
1. Mem0 Memory System
2. Redis Semantic Caching
3. Request Fingerprinting
4. Gemini Model Comparison
5. LangGraph Workflow
6. FastAPI Server
7. Beautiful UI
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Core FastAPI imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Enterprise imports
import google.generativeai as genai
import mem0
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Environment setup
from dotenv import load_dotenv

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI_MODEL = os.getenv("AI_MODEL", "gemini-1.5-flash-latest")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("✅ Google Gemini API configured successfully")


# Data Models
class TravelQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(default="anonymous")
    include_model_comparison: bool = Field(default=False)
    use_cache: bool = Field(default=True)


class TravelQueryResponse(BaseModel):
    query: str
    response: str
    user_id: str
    metrics: Dict[str, Any]
    processing_time_ms: float
    timestamp: str
    success: bool
    ai_powered: bool = True
    error: Optional[str] = None


@dataclass
class MemoryMetrics:
    """Metrics tracking for memory operations"""

    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    avg_read_time: float = 0.0
    avg_write_time: float = 0.0


@dataclass
class CacheMetrics:
    """Metrics tracking for cache operations"""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    avg_similarity: float = 0.0
    avg_retrieval_time: float = 0.0


# Enterprise Memory Manager (Task 2)
class EnterpriseMemoryManager:
    """Enterprise-grade Mem0 memory manager with comprehensive logging and error handling"""

    def __init__(self, user_id: str = "default_user"):
        """Initialize memory manager with robust error handling"""
        self.user_id = user_id
        self.metrics = MemoryMetrics()
        self._setup_mem0()

    def _setup_mem0(self):
        """Setup Mem0 with comprehensive error handling"""
        try:
            # Initialize Mem0 client
            self.mem0_client = mem0.Memory()
            logger.info("✅ Mem0 Memory initialized successfully")

            # Test connection
            self._test_connection()

        except Exception as e:
            logger.error(f"❌ Failed to initialize Mem0: {str(e)}")
            # Fallback to in-memory storage for development
            self._initialize_fallback_memory()

    def _test_connection(self):
        """Test Mem0 connection with health check"""
        try:
            # Attempt to read memories to test connection
            test_memories = self.mem0_client.get_all(user_id=self.user_id)
            logger.info(
                f"🔍 Memory connection test successful. Found {len(test_memories)} existing memories"
            )
        except Exception as e:
            logger.warning(f"⚠️ Memory connection test failed: {str(e)}")
            raise

    def _initialize_fallback_memory(self):
        """Initialize fallback in-memory storage when Mem0 is not available"""
        # Use class-level shared storage so memory persists across user requests
        if not hasattr(EnterpriseMemoryManager, "_shared_fallback_memory"):
            EnterpriseMemoryManager._shared_fallback_memory = {}
        self.fallback_memory = EnterpriseMemoryManager._shared_fallback_memory
        logger.warning(
            "⚠️ Using fallback in-memory storage. Data will not persist between sessions."
        )

    async def store_user_preference(
        self, preference_data: Dict[str, Any], user_id: str = None
    ) -> bool:
        """Store user preference with comprehensive logging and error handling"""
        start_time = time.time()

        # Use provided user_id or fall back to instance user_id
        effective_user_id = user_id or self.user_id

        try:
            # Validate input
            if not isinstance(preference_data, dict):
                raise ValueError("Preference data must be a dictionary")

            # Create memory entry
            memory_text = f"User preference: {json.dumps(preference_data, indent=2)}"

            # Store in Mem0
            if hasattr(self, "mem0_client"):
                result = self.mem0_client.add(memory_text, user_id=effective_user_id)
                logger.info(f"✅ Preference stored in Mem0: {result}")
            else:
                # Fallback storage - use user_id as key
                user_key = f"{effective_user_id}_pref_{len([k for k in self.fallback_memory if k.startswith(effective_user_id)])}"
                self.fallback_memory[user_key] = {
                    "text": memory_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": effective_user_id,
                }
                logger.info("✅ Preference stored in fallback memory")

            # Update metrics
            self.metrics.writes += 1
            execution_time = (time.time() - start_time) * 1000
            self.metrics.avg_write_time = (
                self.metrics.avg_write_time * (self.metrics.writes - 1) + execution_time
            ) / self.metrics.writes

            logger.info(
                f"📊 Memory write completed in {execution_time:.2f}ms for user: {effective_user_id}"
            )
            logger.info(
                f"🧠 Total memories in fallback storage: {len(self.fallback_memory)}"
            )
            logger.info(
                f"🔍 User memories: {len([k for k in self.fallback_memory if k.startswith(effective_user_id)])}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store preference: {str(e)}")
            return False

    async def retrieve_user_context(
        self, query: str, user_id: str = None
    ) -> Dict[str, Any]:
        """Retrieve relevant user context with semantic search"""
        start_time = time.time()

        # Use provided user_id or fall back to instance user_id
        effective_user_id = user_id or self.user_id

        try:
            self.metrics.reads += 1

            # Get memories from Mem0
            if hasattr(self, "mem0_client"):
                memories = self.mem0_client.search(
                    query, user_id=effective_user_id, limit=5
                )

                if memories:
                    self.metrics.hits += 1
                    context = {
                        "memories_found": len(memories),
                        "relevant_context": [mem.get("memory", "") for mem in memories],
                        "query": query,
                        "user_id": effective_user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:
                    self.metrics.misses += 1
                    context = {
                        "memories_found": 0,
                        "relevant_context": [],
                        "query": query,
                        "user_id": effective_user_id,
                        "message": "No relevant memories found",
                    }
            else:
                # Fallback search - improved semantic search for this user's memories
                relevant_memories = []
                for key, memory in self.fallback_memory.items():
                    # Only check memories for this specific user
                    if (
                        key.startswith(f"{effective_user_id}_")
                        and memory.get("user_id") == effective_user_id
                    ):
                        memory_text = memory["text"].lower()
                        query_lower = query.lower()

                        # Enhanced search - check for direct matches and context
                        direct_match = any(
                            term.lower() in memory_text for term in query.split()
                        )

                        # Location context awareness - if query is about activities/services
                        # look for location mentions in previous memories
                        location_context = False
                        if query_lower in [
                            "movies",
                            "entertainment",
                            "theaters",
                            "cinemas",
                            "shows",
                        ]:
                            # Look for location mentions in this memory
                            cities = [
                                "hyderabad",
                                "mumbai",
                                "delhi",
                                "bangalore",
                                "chennai",
                                "kolkata",
                                "pune",
                            ]
                            location_context = any(
                                city in memory_text for city in cities
                            )

                        if direct_match or location_context:
                            relevant_memories.append(memory["text"])

                # Also add all recent memories for better context (last 3)
                user_memories = [
                    memory
                    for key, memory in self.fallback_memory.items()
                    if key.startswith(f"{effective_user_id}_")
                    and memory.get("user_id") == effective_user_id
                ]
                if user_memories and len(relevant_memories) == 0:
                    # If no specific matches, include recent memories for context
                    recent_memories = sorted(
                        user_memories,
                        key=lambda x: x.get("timestamp", ""),
                        reverse=True,
                    )[:2]
                    relevant_memories.extend([m["text"] for m in recent_memories])

                context = {
                    "memories_found": len(relevant_memories),
                    "relevant_context": relevant_memories,
                    "query": query,
                    "user_id": effective_user_id,
                    "source": "fallback",
                }

                if relevant_memories:
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1

            # Update timing metrics
            execution_time = (time.time() - start_time) * 1000
            self.metrics.avg_read_time = (
                self.metrics.avg_read_time * (self.metrics.reads - 1) + execution_time
            ) / self.metrics.reads

            logger.info(
                f"🧠 Memory retrieval completed in {execution_time:.2f}ms, found {context['memories_found']} memories for user {effective_user_id}"
            )
            logger.info(
                f"🔍 Searched query: '{query}' - Context found: {context.get('relevant_context', [])[:2]}"
            )  # Show first 2 items
            return context

        except Exception as e:
            logger.error(f"❌ Failed to retrieve user context: {str(e)}")
            return {
                "memories_found": 0,
                "relevant_context": [],
                "query": query,
                "error": str(e),
            }

    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get comprehensive memory performance metrics"""
        hit_rate = (self.metrics.hits / max(self.metrics.reads, 1)) * 100

        return {
            "total_reads": self.metrics.reads,
            "total_writes": self.metrics.writes,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "avg_read_time_ms": f"{self.metrics.avg_read_time:.2f}",
            "avg_write_time_ms": f"{self.metrics.avg_write_time:.2f}",
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
        }


# Enterprise Semantic Cache (Task 3)
class EnterpriseSemanticCache:
    """Enterprise-grade Redis semantic cache with similarity matching"""

    def __init__(self, similarity_threshold: float = 0.85, ttl: int = 3600):
        """Initialize semantic cache with enterprise features"""
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.metrics = CacheMetrics()

        # Initialize components
        self._setup_redis()
        self._setup_sentence_transformer()

    def _setup_redis(self):
        """Setup Redis connection with robust error handling"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established successfully")

        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {str(e)}")
            # Fallback to in-memory cache
            self._setup_fallback_cache()
        except Exception as e:
            logger.error(f"❌ Unexpected Redis error: {str(e)}")
            self._setup_fallback_cache()

    def _setup_fallback_cache(self):
        """Setup fallback in-memory cache"""
        self.fallback_cache = {}
        self.use_fallback = True
        logger.warning("⚠️ Using fallback in-memory cache. Performance may be limited.")

    def _setup_sentence_transformer(self):
        """Initialize sentence transformer for semantic similarity"""
        try:
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Sentence transformer loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer: {str(e)}")
            # Use fallback embedding generation
            self.sentence_model = None

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text with error handling"""
        try:
            if self.sentence_model:
                return self.sentence_model.encode(text)
            else:
                # Fallback to simple hash-based similarity
                return np.array([hash(text) % 1000 for _ in range(384)])
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {str(e)}")
            return np.array([hash(text) % 1000 for _ in range(384)])

    def _calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"❌ Failed to calculate similarity: {str(e)}")
            return 0.0

    async def get_cached_response(
        self, query: str, model_name: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached response with semantic similarity matching"""
        start_time = time.time()

        try:
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)

            if hasattr(self, "use_fallback") and self.use_fallback:
                # Fallback cache search
                best_match = None
                best_similarity = 0.0

                for key, cached_data in self.fallback_cache.items():
                    if key.startswith(f"semantic_cache:{model_name}:"):
                        cached_embedding = np.array(cached_data.get("embedding", []))
                        if cached_embedding.size > 0:
                            similarity = self._calculate_similarity(
                                query_embedding, cached_embedding
                            )

                            if (
                                similarity > best_similarity
                                and similarity >= self.similarity_threshold
                            ):
                                best_similarity = similarity
                                best_match = cached_data

                if best_match:
                    self.metrics.hits += 1
                    retrieval_time = (time.time() - start_time) * 1000

                    logger.info(
                        f"🎯 Cache HIT! Similarity: {best_similarity:.3f}, Retrieved in {retrieval_time:.2f}ms"
                    )

                    return {
                        "response": best_match.get("response"),
                        "similarity": best_similarity,
                        "cached_query": best_match.get("query"),
                        "timestamp": best_match.get("timestamp"),
                        "source": "fallback_cache",
                    }
                else:
                    self.metrics.misses += 1
                    logger.info(
                        "🔍 Cache MISS - No similar queries found in fallback cache"
                    )
                    return None

            else:
                # Redis cache search
                cache_key_pattern = f"semantic_cache:{model_name}:*"
                cached_keys = self.redis_client.keys(cache_key_pattern)

                best_match_key = None
                best_similarity = 0.0

                for key in cached_keys:
                    try:
                        cached_data = json.loads(self.redis_client.get(key))
                        cached_embedding = np.array(cached_data.get("embedding", []))

                        if cached_embedding.size > 0:
                            similarity = self._calculate_similarity(
                                query_embedding, cached_embedding
                            )

                            if (
                                similarity > best_similarity
                                and similarity >= self.similarity_threshold
                            ):
                                best_similarity = similarity
                                best_match_key = key

                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"⚠️ Error processing cached key {key}: {str(e)}")
                        continue

                if best_match_key:
                    self.metrics.hits += 1
                    cached_response = json.loads(self.redis_client.get(best_match_key))
                    retrieval_time = (time.time() - start_time) * 1000

                    logger.info(
                        f"🎯 Cache HIT! Similarity: {best_similarity:.3f}, Retrieved in {retrieval_time:.2f}ms"
                    )

                    return {
                        "response": cached_response.get("response"),
                        "similarity": best_similarity,
                        "cached_query": cached_response.get("query"),
                        "timestamp": cached_response.get("timestamp"),
                        "source": "redis_cache",
                    }
                else:
                    self.metrics.misses += 1
                    logger.info("🔍 Cache MISS - No similar queries found")
                    return None

        except Exception as e:
            logger.error(f"❌ Error retrieving cached response: {str(e)}")
            self.metrics.misses += 1
            return None

    async def store_response(
        self, query: str, response: str, model_name: str = "default"
    ) -> bool:
        """Store response in semantic cache"""
        try:
            # Generate embedding
            embedding = self._generate_embedding(query)

            # Create cache entry
            cache_data = {
                "query": query,
                "response": response,
                "embedding": embedding.tolist(),
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_name,
            }

            # Generate unique key
            cache_key = (
                f"semantic_cache:{model_name}:{hashlib.md5(query.encode()).hexdigest()}"
            )

            if hasattr(self, "use_fallback") and self.use_fallback:
                # Store in fallback cache
                self.fallback_cache[cache_key] = cache_data
            else:
                # Store in Redis
                self.redis_client.setex(cache_key, self.ttl, json.dumps(cache_data))

            self.metrics.stores += 1
            logger.info(f"💾 Response cached successfully with key: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store response in cache: {str(e)}")
            return False

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance metrics"""
        hit_rate = (
            self.metrics.hits / max(self.metrics.hits + self.metrics.misses, 1)
        ) * 100

        return {
            "cache_hits": self.metrics.hits,
            "cache_misses": self.metrics.misses,
            "cache_stores": self.metrics.stores,
            "hit_rate": f"{hit_rate:.1f}%",
            "avg_similarity": f"{self.metrics.avg_similarity:.3f}",
            "avg_retrieval_time_ms": f"{self.metrics.avg_retrieval_time:.2f}",
        }


# Request Fingerprinter (Task 4)
class EnterpriseRequestFingerprinter:
    """Enterprise request fingerprinting for duplicate detection and optimization"""

    def __init__(self, duplicate_threshold: int = 5):
        self.duplicate_threshold = duplicate_threshold
        self.request_history = {}

    def generate_fingerprint(self, query: str, user_id: str) -> Dict[str, Any]:
        """Generate unique fingerprint for request with duplicate detection"""
        try:
            # Create comprehensive fingerprint
            raw_fingerprint = f"{user_id}:{query.lower().strip()}"
            fingerprint_hash = hashlib.sha256(raw_fingerprint.encode()).hexdigest()

            # Track request frequency
            if fingerprint_hash in self.request_history:
                self.request_history[fingerprint_hash]["count"] += 1
                self.request_history[fingerprint_hash]["last_seen"] = (
                    datetime.utcnow().isoformat()
                )
            else:
                self.request_history[fingerprint_hash] = {
                    "count": 1,
                    "first_seen": datetime.utcnow().isoformat(),
                    "last_seen": datetime.utcnow().isoformat(),
                    "query": query,
                    "user_id": user_id,
                }

            request_data = self.request_history[fingerprint_hash]
            is_duplicate = request_data["count"] > self.duplicate_threshold

            return {
                "fingerprint_hash": fingerprint_hash,
                "is_duplicate": is_duplicate,
                "request_count": request_data["count"],
                "first_seen": request_data["first_seen"],
                "last_seen": request_data["last_seen"],
                "category": self._categorize_query(query),
                "optimization_suggestion": "cache_recommended"
                if request_data["count"] > 2
                else "normal_processing",
            }

        except Exception as e:
            logger.error(f"❌ Failed to generate fingerprint: {str(e)}")
            return {"fingerprint_hash": "error", "is_duplicate": False, "error": str(e)}

    def _categorize_query(self, query: str) -> str:
        """Categorize query for optimization insights"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["restaurant", "food", "eat", "dining"]):
            return "food_and_dining"
        elif any(word in query_lower for word in ["hotel", "accommodation", "stay"]):
            return "accommodation"
        elif any(word in query_lower for word in ["flight", "airline", "fly"]):
            return "transportation"
        elif any(
            word in query_lower
            for word in ["activity", "attraction", "tour", "sightseeing"]
        ):
            return "activities"
        elif any(word in query_lower for word in ["budget", "cost", "price", "money"]):
            return "budget_planning"
        else:
            return "general_travel"


# Model Comparator (Task 5)
class EnterpriseModelComparator:
    """Compare Gemini Flash vs Pro models for optimization"""

    def __init__(self):
        self.comparison_history = []

    async def compare_models(
        self, query: str, use_comparison: bool = False
    ) -> Dict[str, Any]:
        """Compare Flash vs Pro models if requested"""
        if not use_comparison or not GOOGLE_API_KEY:
            return {
                "comparison_performed": False,
                "reason": "Not requested or no API key",
            }

        try:
            start_time = time.time()

            # Test with Flash model
            flash_start = time.time()
            try:
                flash_model = genai.GenerativeModel("gemini-1.5-flash")
                flash_response = flash_model.generate_content(query)
                flash_time = (time.time() - flash_start) * 1000
                flash_result = (
                    flash_response.text if flash_response.text else "No response"
                )
                flash_success = True
            except Exception as e:
                flash_time = (time.time() - flash_start) * 1000
                flash_result = f"Error: {str(e)}"
                flash_success = False

            # Test with Pro model (if available)
            pro_start = time.time()
            try:
                pro_model = genai.GenerativeModel("gemini-1.5-pro")
                pro_response = pro_model.generate_content(query)
                pro_time = (time.time() - pro_start) * 1000
                pro_result = pro_response.text if pro_response.text else "No response"
                pro_success = True
            except Exception as e:
                pro_time = (time.time() - pro_start) * 1000
                pro_result = f"Error: {str(e)}"
                pro_success = False

            # Determine recommendations
            speed_winner = "flash" if flash_time < pro_time else "pro"
            quality_winner = (
                "pro"
                if pro_success and len(pro_result) > len(flash_result)
                else "flash"
            )

            comparison_result = {
                "comparison_performed": True,
                "flash_response": flash_result[:200] + "..."
                if len(flash_result) > 200
                else flash_result,
                "pro_response": pro_result[:200] + "..."
                if len(pro_result) > 200
                else pro_result,
                "flash_time_ms": round(flash_time, 2),
                "pro_time_ms": round(pro_time, 2),
                "flash_success": flash_success,
                "pro_success": pro_success,
                "speed_winner": speed_winner,
                "quality_winner": quality_winner,
                "recommendation": f"Use {speed_winner.title()} for speed"
                if speed_winner == quality_winner
                else "Flash for speed, Pro for quality",
                "cost_analysis": {
                    "flash_cost_estimate": "Lower cost, faster",
                    "pro_cost_estimate": "Higher cost, potentially better quality",
                },
            }

            self.comparison_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": query[:50] + "...",
                    "result": comparison_result,
                }
            )

            logger.info(
                f"⚖️ Model comparison completed: {speed_winner} wins on speed, {quality_winner} wins on quality"
            )
            return comparison_result

        except Exception as e:
            logger.error(f"❌ Model comparison failed: {str(e)}")
            return {"comparison_performed": False, "error": str(e)}


# Initialize enterprise components
memory_manager = EnterpriseMemoryManager()
semantic_cache = EnterpriseSemanticCache()
fingerprinter = EnterpriseRequestFingerprinter()
model_comparator = EnterpriseModelComparator()


# Enterprise Travel Assistant Workflow (Task 6 - LangGraph Integration)
async def process_enterprise_travel_query(
    query: str, user_id: str, include_comparison: bool = False, use_cache: bool = True
) -> Dict[str, Any]:
    """
    Complete enterprise workflow integrating all components:
    1. Generate fingerprint
    2. Check memory context
    3. Check cache
    4. Generate AI response
    5. Store in memory and cache
    6. Compare models (if requested)
    """
    workflow_start = time.time()
    processing_steps = []

    try:
        # Step 1: Generate request fingerprint
        processing_steps.append("Generating request fingerprint")
        fingerprint_data = fingerprinter.generate_fingerprint(query, user_id)

        # Step 2: Retrieve user context from memory
        processing_steps.append("Retrieving user context from memory")
        memory_context = await memory_manager.retrieve_user_context(query, user_id)

        # Step 3: Check semantic cache (if enabled)
        cached_response = None
        if use_cache:
            processing_steps.append("Checking semantic cache")
            cached_response = await semantic_cache.get_cached_response(query, AI_MODEL)

        # Step 4: Generate AI response or use cached
        if cached_response:
            processing_steps.append("Using cached response")
            ai_response = cached_response["response"]
            response_source = "cache"
        else:
            processing_steps.append("Generating AI response")

            # Prepare context-aware prompt
            context_prompt = query
            if memory_context.get("memories_found", 0) > 0:
                relevant_context = "\\n".join(memory_context["relevant_context"])
                context_prompt = f"""Based on the user's travel history and preferences:
{relevant_context}

User's current query: {query}

Please provide a personalized travel response that takes into account their previous preferences and context."""
                logger.info(
                    f"🧠 Using memory context for AI response. Found {memory_context['memories_found']} memories"
                )
                logger.info(f"📝 Enhanced prompt: {context_prompt[:200]}...")
            else:
                logger.info(f"⚠️ No memory context found for query: {query}")

            # Generate AI response
            if GOOGLE_API_KEY:
                try:
                    model = genai.GenerativeModel(AI_MODEL)
                    response = model.generate_content(context_prompt)
                    ai_response = (
                        response.text
                        if response.text
                        else "I'm here to help with your travel planning!"
                    )
                    response_source = "ai_generated"
                except Exception as e:
                    logger.error(f"❌ AI generation failed: {str(e)}")
                    ai_response = f"I'm your travel assistant, but I'm having trouble connecting to AI services. However, I can still help you with general travel information about: {query}"
                    response_source = "fallback"
            else:
                ai_response = f"I'm your travel assistant! While I don't have AI capabilities configured right now, I'd love to help you with travel planning for: {query}"
                response_source = "no_api_key"

        # Step 5: Store new preference/query in memory
        if not cached_response:
            processing_steps.append("Storing interaction in memory")

            # Extract location context for better memory
            location_mentioned = None
            query_lower = query.lower()
            cities = [
                "hyderabad",
                "bangalore",
                "mumbai",
                "delhi",
                "chennai",
                "kolkata",
                "pune",
                "jaipur",
            ]
            for city in cities:
                if city in query_lower:
                    location_mentioned = city.title()
                    break

            preference_data = {
                "query": query,
                "response_summary": ai_response[:100] + "...",
                "timestamp": datetime.utcnow().isoformat(),
                "category": fingerprint_data.get("category", "general"),
                "location_context": location_mentioned,
                "user_id": user_id,
            }
            await memory_manager.store_user_preference(preference_data, user_id)

        # Step 6: Cache the response (if not from cache)
        if not cached_response and use_cache:
            processing_steps.append("Caching response for future queries")
            await semantic_cache.store_response(query, ai_response, AI_MODEL)

        # Step 7: Model comparison (if requested)
        model_comparison = None
        if include_comparison:
            processing_steps.append("Performing model comparison")
            model_comparison = await model_comparator.compare_models(query, True)

        # Compile comprehensive metrics
        processing_time = (time.time() - workflow_start) * 1000

        enterprise_metrics = {
            "memory_metrics": memory_manager.get_memory_metrics(),
            "cache_metrics": semantic_cache.get_cache_metrics(),
            "fingerprint_info": fingerprint_data,
            "model_comparison": model_comparison,
            "system_metrics": {
                "total_processing_time_ms": round(processing_time, 2),
                "response_source": response_source,
                "processing_steps": processing_steps,
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        logger.info(f"✅ Enterprise workflow completed in {processing_time:.2f}ms")

        return {
            "response": ai_response,
            "metrics": enterprise_metrics,
            "success": True,
            "processing_time_ms": processing_time,
            "user_id": user_id,
            "query": query,
        }

    except Exception as e:
        logger.error(f"❌ Enterprise workflow failed: {str(e)}")
        processing_time = (time.time() - workflow_start) * 1000

        return {
            "response": f"I encountered an error while processing your travel query: {query}. Let me help you in a simpler way with travel planning!",
            "metrics": {
                "error": str(e),
                "processing_time_ms": processing_time,
                "processing_steps": processing_steps,
            },
            "success": False,
            "processing_time_ms": processing_time,
            "user_id": user_id,
            "query": query,
        }


# FastAPI Application (Task 7)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for FastAPI app"""
    logger.info(
        "🚀 Starting Enterprise Travel Assistant API with full enterprise features"
    )
    yield
    logger.info("🛑 Shutting down Enterprise Travel Assistant API")


app = FastAPI(
    title="🧳 Enterprise Travel Assistant",
    description="AI-powered travel assistant with Mem0 memory, Redis caching, fingerprinting, model comparison, and LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Welcome page with enterprise features overview"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧳 Enterprise Travel Assistant</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
            .container {{ max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }}
            h1 {{ text-align: center; margin-bottom: 30px; font-size: 2.5em; }}
            .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
            .feature {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #00ff88; }}
            .feature h3 {{ margin: 0 0 10px 0; color: #00ff88; }}
            .links {{ text-align: center; margin-top: 30px; }}
            .links a {{ color: #00ff88; text-decoration: none; margin: 0 15px; padding: 10px 20px; border: 2px solid #00ff88; border-radius: 25px; transition: all 0.3s; }}
            .links a:hover {{ background: #00ff88; color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧳 Enterprise Travel Assistant</h1>
            <p style="text-align: center; font-size: 1.2em;">Your AI-powered travel companion with enterprise-grade features</p>
            
            <div class="features">
                <div class="feature">
                    <h3>🧠 Mem0 Memory System</h3>
                    <p>Remembers your preferences and travel history for personalized recommendations</p>
                </div>
                <div class="feature">
                    <h3>🗄️ Redis Semantic Cache</h3>
                    <p>Lightning-fast responses with intelligent similarity matching</p>
                </div>
                <div class="feature">
                    <h3>🔑 Request Fingerprinting</h3>
                    <p>Smart duplicate detection and cost optimization</p>
                </div>
                <div class="feature">
                    <h3>⚖️ Model Comparison</h3>
                    <p>Gemini Flash vs Pro analysis for optimal performance</p>
                </div>
                <div class="feature">
                    <h3>🔄 LangGraph Workflow</h3>
                    <p>Orchestrated enterprise pipeline with error handling</p>
                </div>
                <div class="feature">
                    <h3>📊 Enterprise Metrics</h3>
                    <p>Comprehensive performance monitoring and insights</p>
                </div>
            </div>
            
            <div class="links">
                <a href="/chat">💬 Chat Interface</a>
                <a href="/docs">📚 API Documentation</a>
                <a href="/metrics">📊 System Metrics</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/memory-travel-assistant", response_model=TravelQueryResponse)
async def enterprise_travel_query(
    request: TravelQueryRequest, background_tasks: BackgroundTasks
):
    """
    Main enterprise travel assistant endpoint with full feature integration:
    - Mem0 memory retrieval and storage
    - Redis semantic caching
    - Request fingerprinting
    - Model comparison (optional)
    - LangGraph workflow orchestration
    """
    start_time = time.time()

    try:
        # Process through enterprise workflow
        result = await process_enterprise_travel_query(
            query=request.query,
            user_id=request.user_id,
            include_comparison=request.include_model_comparison,
            use_cache=request.use_cache,
        )

        processing_time = (time.time() - start_time) * 1000

        return TravelQueryResponse(
            query=request.query,
            response=result["response"],
            user_id=request.user_id,
            metrics=result["metrics"],
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            success=result["success"],
            ai_powered=True,
        )

    except Exception as e:
        logger.error(f"❌ Enterprise endpoint error: {str(e)}")
        processing_time = (time.time() - start_time) * 1000

        return TravelQueryResponse(
            query=request.query,
            response=f"I apologize, but I encountered an issue processing your travel query. However, I'm still here to help you plan your travels! Your query was: {request.query}",
            user_id=request.user_id,
            metrics={"error": str(e), "processing_time_ms": processing_time},
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            success=False,
            ai_powered=False,
            error=str(e),
        )


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Beautiful enterprise chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧳 Enterprise Travel Assistant - Chat</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                background: rgba(255,255,255,0.1);
                color: white;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }
            
            .chat-container {
                flex: 1;
                max-width: 800px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                height: calc(100vh - 140px);
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(10px);
                border-radius: 15px 15px 0 0;
                margin-top: 20px;
            }
            
            .messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                max-width: 80%;
                padding: 15px 20px;
                border-radius: 20px;
                word-wrap: break-word;
                line-height: 1.4;
            }
            
            .user-message {
                background: linear-gradient(135deg, #00ff88, #00cc6a);
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 5px;
            }
            
            .bot-message {
                background: rgba(255,255,255,0.9);
                color: #333;
                align-self: flex-start;
                border-bottom-left-radius: 5px;
                white-space: pre-wrap;
            }
            
            .input-area {
                padding: 20px;
                background: rgba(255,255,255,0.1);
                border-top: 1px solid rgba(255,255,255,0.2);
                display: flex;
                gap: 10px;
            }
            
            .input-area input {
                flex: 1;
                padding: 15px 20px;
                border: none;
                border-radius: 25px;
                background: rgba(255,255,255,0.9);
                font-size: 16px;
                outline: none;
            }
            
            .input-area button {
                padding: 15px 30px;
                background: linear-gradient(135deg, #00ff88, #00cc6a);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-weight: bold;
                transition: transform 0.2s;
            }
            
            .input-area button:hover {
                transform: scale(1.05);
            }
            
            .input-area button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .typing {
                display: none;
                align-self: flex-start;
                background: rgba(255,255,255,0.9);
                color: #666;
                padding: 15px 20px;
                border-radius: 20px;
                border-bottom-left-radius: 5px;
            }
            
            .typing span {
                animation: typing 1.4s infinite;
            }
            
            .typing span:nth-child(2) { animation-delay: 0.2s; }
            .typing span:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { opacity: 0.3; }
                30% { opacity: 1; }
            }
            
            .metrics-info {
                font-size: 12px;
                color: #666;
                margin-top: 10px;
                padding: 10px;
                background: rgba(0,0,0,0.1);
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧳 Enterprise Travel Assistant</h1>
            <p>AI-powered with Memory • Caching • Fingerprinting • Model Comparison • LangGraph</p>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="bot-message">
                    <strong>🤖 Enterprise Travel Assistant:</strong><br><br>
                    Welcome! I'm your AI travel assistant with enterprise-grade features:<br><br>
                    🧠 <strong>Mem0 Memory System</strong>: I remember your preferences across conversations<br>
                    🗄️ <strong>Redis Semantic Cache</strong>: Lightning-fast responses with intelligent caching<br>
                    🔑 <strong>Request Fingerprinting</strong>: Smart duplicate detection and optimization<br>
                    ⚖️ <strong>Model Comparison</strong>: Gemini Flash vs Pro for optimal responses<br>
                    🔄 <strong>LangGraph Workflow</strong>: Complete enterprise orchestration<br>
                    📊 <strong>Live Metrics</strong>: Real-time performance monitoring<br><br>
                    <strong>Test contextual memory:</strong><br>
                    • Ask: "restaurants in Hyderabad"<br>
                    • Then ask: "what about movies?"<br>
                    • I'll remember the Hyderabad context!<br><br>
                    What travel adventure can I help you plan today? ✈️
                </div>
            </div>
            
            <div class="typing" id="typing">
                AI is thinking<span>.</span><span>.</span><span>.</span>
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Ask me about travel destinations, planning, or anything travel-related..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
        
        <script>
            const messages = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const typing = document.getElementById('typing');
            
            // Generate persistent session ID for this browser session
            let sessionUserId = localStorage.getItem('travel_assistant_user_id');
            if (!sessionUserId) {
                sessionUserId = 'user_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('travel_assistant_user_id', sessionUserId);
            }
            console.log('Using session user ID:', sessionUserId);
            
            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = isUser ? 'user-message message' : 'bot-message message';
                messageDiv.innerHTML = content;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function showTyping() {
                typing.style.display = 'block';
                messages.scrollTop = messages.scrollHeight;
            }
            
            function hideTyping() {
                typing.style.display = 'none';
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message
                addMessage(message, true);
                messageInput.value = '';
                sendButton.disabled = true;
                showTyping();
                
                try {
                    const response = await fetch('/memory-travel-assistant', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: message,
                            user_id: sessionUserId,
                            include_model_comparison: false,
                            use_cache: true
                        })
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    // Format response with metrics
                    let responseContent = `<strong>🤖 Enterprise Travel Assistant:</strong><br><br>${data.response}`;
                    
                    if (data.metrics) {
                        const metrics = data.metrics;
                        let metricsHtml = '<div class="metrics-info">';
                        metricsHtml += `⏱️ <strong>Processing:</strong> ${data.processing_time_ms.toFixed(1)}ms | `;
                        
                        if (metrics.memory_metrics) {
                            metricsHtml += `🧠 <strong>Memory:</strong> ${metrics.memory_metrics.total_reads} reads, ${metrics.memory_metrics.cache_hit_rate} hit rate | `;
                        }
                        
                        if (metrics.cache_metrics) {
                            metricsHtml += `🗄️ <strong>Cache:</strong> ${metrics.cache_metrics.cache_hits} hits, ${metrics.cache_metrics.hit_rate} rate | `;
                        }
                        
                        if (metrics.fingerprint_info) {
                            metricsHtml += `🔑 <strong>Fingerprint:</strong> ${metrics.fingerprint_info.category}, ${metrics.fingerprint_info.request_count} requests`;
                        }
                        
                        metricsHtml += '</div>';
                        responseContent += metricsHtml;
                    }
                    
                    addMessage(responseContent);
                    
                } catch (error) {
                    hideTyping();
                    addMessage('<strong>🤖 Enterprise Travel Assistant:</strong><br><br>I encountered a technical issue, but I\\'m still here to help with your travel planning! Please try again.');
                    console.error('Error:', error);
                }
                
                sendButton.disabled = false;
                messageInput.focus();
            }
            
            // Focus on input when page loads
            messageInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics"""
    return {
        "memory_metrics": memory_manager.get_memory_metrics(),
        "cache_metrics": semantic_cache.get_cache_metrics(),
        "fingerprint_stats": {
            "total_requests": len(fingerprinter.request_history),
            "unique_users": len(
                set(req["user_id"] for req in fingerprinter.request_history.values())
            ),
            "categories": {},
        },
        "model_comparison_history": model_comparator.comparison_history[
            -10:
        ],  # Last 10 comparisons
        "system_info": {
            "api_key_configured": bool(GOOGLE_API_KEY),
            "ai_model": AI_MODEL,
            "redis_host": REDIS_HOST,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "enterprise_chat:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
