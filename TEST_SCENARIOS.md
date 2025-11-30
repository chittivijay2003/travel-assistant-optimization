# Testing Scenarios for Travel Assistant

## 1. Location Context Management

### Scenario 1.1: Single Location Queries
**Test Case**: User asks about one location consistently
```
Query 1: "restaurants in Hyderabad"
Expected: Hyderabad restaurants (Paradise Biryani, Shah Ghouse, etc.)

Query 2: "movie theaters"
Expected: Should infer Hyderabad from context, show Hyderabad theaters

Query 3: "hotels"
Expected: Should continue with Hyderabad context
```

### Scenario 1.2: Location Switch (CRITICAL)
**Test Case**: User switches between different locations
```
Query 1: "restaurants in Hyderabad"
Expected: Hyderabad restaurants

Query 2: "movie theaters in Goa"
Expected: GOA theaters (INOX Panjim, Margao) - NOT Hyderabad

Query 3: "beaches"
Expected: Should infer Goa from Query 2, show Goa beaches

Query 4: "restaurants in Mumbai"
Expected: Mumbai restaurants - NOT Hyderabad or Goa
```

### Scenario 1.3: Ambiguous Location
**Test Case**: User query without location after multi-city discussion
```
Query 1: "restaurants in Delhi"
Expected: Delhi restaurants

Query 2: "beaches in Goa"
Expected: Goa beaches

Query 3: "restaurants"
Expected: AI should ask for clarification OR infer most recent location (Goa)
```

### Scenario 1.4: Location Variations
**Test Case**: Different ways of specifying location
```
Query 1: "beaches near Mumbai"
Expected: Mumbai beaches

Query 2: "hotels at Bangalore"
Expected: Bangalore hotels

Query 3: "restaurants around Chennai"
Expected: Chennai restaurants

Query 4: "things to do in Kerala"
Expected: Kerala attractions
```

---

## 2. Memory Context & Personalization

### Scenario 2.1: Preference Learning
**Test Case**: User shares preferences, system remembers
```
Query 1: "I love spicy food and vegetarian cuisine"
Expected: Acknowledgment, memory stored

Query 2: "restaurants in Hyderabad"
Expected: Should recommend vegetarian restaurants with spicy options

Query 3: "restaurants in Mumbai"
Expected: Should STILL remember vegetarian + spicy preference for Mumbai
```

### Scenario 2.2: Budget Preference
**Test Case**: User specifies budget constraints
```
Query 1: "I'm looking for budget-friendly options under $20 per person"
Expected: Acknowledgment, memory stored

Query 2: "hotels in Goa"
Expected: Should recommend budget hotels

Query 3: "restaurants in Hyderabad"
Expected: Should recommend affordable restaurants
```

### Scenario 2.3: Travel Style
**Test Case**: User shares travel preferences
```
Query 1: "I prefer adventure activities and nature over city tours"
Expected: Acknowledgment, memory stored

Query 2: "things to do in Manali"
Expected: Should recommend trekking, paragliding, nature spots

Query 3: "what about in Goa?"
Expected: Should recommend water sports, hiking, NOT nightclubs/casinos
```

---

## 3. Semantic Caching

### Scenario 3.1: Similar Query Caching
**Test Case**: Similar queries should use cache
```
Query 1: "beaches in Goa"
Expected: Full AI response (16-28s latency)

Query 2: "show me beaches in Goa"
Expected: Should use cached response (< 1s latency)

Query 3: "what are the best beaches in Goa"
Expected: Should use cached response (similarity > 0.85)
```

### Scenario 3.2: Different User Same Query
**Test Case**: Different users asking same question
```
User A Query: "restaurants in Mumbai"
Expected: Full AI response, cached with user_a ID

User B Query: "restaurants in Mumbai"
Expected: Should generate NEW response (different user_id)
```

### Scenario 3.3: Cache Invalidation
**Test Case**: Significantly different query
```
Query 1: "beaches in Goa"
Expected: Full AI response

Query 2: "mountain trekking in Himachal"
Expected: Full AI response (no cache hit, different topic)
```

---

## 4. Model Comparison (Flash vs Pro)

### Scenario 4.1: Response Quality
**Test Case**: Compare Flash and Pro responses
```
Query: "Recommend a 3-day itinerary for Jaipur"
Expected:
- Flash: Faster response (15-20s), good quality
- Pro: Slower response (25-35s), more detailed
- Both should provide valid itineraries
```

### Scenario 4.2: Speed vs Detail Trade-off
**Test Case**: Complex query requiring detailed response
```
Query: "Compare adventure activities in Rishikesh vs Manali with safety tips and costs"
Expected:
- Flash: Quick overview
- Pro: More comprehensive comparison
- User can see latency difference in workflow_logs
```

---

## 5. Multi-Turn Conversations

### Scenario 5.1: Follow-up Questions
**Test Case**: User asks follow-up without repeating context
```
Query 1: "restaurants in Hyderabad"
Expected: Hyderabad restaurants

Query 2: "what about vegetarian options?"
Expected: Should filter Hyderabad restaurants for vegetarian

Query 3: "and budget-friendly ones?"
Expected: Should show budget vegetarian restaurants in Hyderabad
```

### Scenario 5.2: Context Switching Mid-Conversation
**Test Case**: User changes topic mid-conversation
```
Query 1: "beaches in Goa"
Expected: Goa beaches

Query 2: "actually, I want to visit mountains instead"
Expected: Should ask which mountain destination

Query 3: "Manali"
Expected: Should switch to Manali recommendations
```

---

## 6. Edge Cases

### Scenario 6.1: No Location Specified Initially
**Test Case**: User doesn't specify location
```
Query 1: "I want to visit beaches"
Expected: AI should ask which location

Query 2: "Goa"
Expected: Should show Goa beaches, remember location
```

### Scenario 6.2: Multiple Locations in One Query
**Test Case**: User asks about multiple cities
```
Query: "Compare beaches in Goa vs Kerala"
Expected: Should provide comparison of both locations
```

### Scenario 6.3: Invalid/Unknown Location
**Test Case**: User asks about non-tourist location
```
Query: "beaches in Lucknow"
Expected: Should politely indicate no beaches in Lucknow, suggest alternatives
```

### Scenario 6.4: Very Long Query
**Test Case**: User provides detailed requirements
```
Query: "I'm planning a 10-day trip to Rajasthan with my family including kids aged 5 and 8. We love history, culture, and want kid-friendly activities. Budget is moderate. Need hotel recommendations, must-visit places, food spots, and safety tips."
Expected: Comprehensive response covering all aspects
```

---

## 7. Memory Retrieval

### Scenario 7.1: Long-term Memory
**Test Case**: User returns after multiple queries
```
Session 1:
Query 1: "I love spicy food"
Query 2: "I prefer budget options"

Session 2 (same user_id):
Query 3: "restaurants in Delhi"
Expected: Should remember spicy + budget preferences from Session 1
```

### Scenario 7.2: Conflicting Preferences
**Test Case**: User changes preferences over time
```
Query 1: "I love non-vegetarian food"
Expected: Memory stored

Query 2: "Actually, I'm vegetarian now"
Expected: Should update memory, use new preference

Query 3: "restaurants in Mumbai"
Expected: Should recommend VEGETARIAN restaurants
```

---

## 8. Workflow Logs Verification

### Scenario 8.1: First Query (No Memory)
**Test Case**: New user, no previous context
```
Query: "beaches in Goa"
Expected workflow_logs:
- has_memory_context: false
- memory_retrieved: []
- flash_response: Present
- pro_response: Present
- destinations: ["Calangute Beach", "Baga Beach", ...]
- memory_updated: true
- cached_fingerprint: Generated
```

### Scenario 8.2: Second Query (With Memory)
**Test Case**: Returning user
```
Query: "hotels in Goa"
Expected workflow_logs:
- has_memory_context: true
- memory_retrieved: ["[Goa] Query: beaches in Goa..."]
- flash_response: Present
- pro_response: Present
- destinations: ["Hotel names..."]
- memory_updated: true
```

---

## 9. API Response Structure

### Scenario 9.1: Valid Response Format
**Test Case**: Check JSON response structure
```
Expected fields:
- query: string
- response: string
- destinations: array
- flash_response: string
- pro_response: string
- flash_latency_ms: number
- pro_latency_ms: number
- faster_model: "flash" or "pro"
- has_memory_context: boolean
- workflow_logs: object
```

---

## 10. Performance Testing

### Scenario 10.1: Concurrent Requests
**Test Case**: Multiple users sending requests simultaneously
```
User A: "beaches in Goa"
User B: "restaurants in Delhi"
User C: "hotels in Mumbai"
Expected: All should get correct responses without mixing context
```

### Scenario 10.2: Rapid Sequential Queries
**Test Case**: Same user sending rapid queries
```
Query 1: "beaches in Goa" (immediate)
Query 2: "hotels in Goa" (1 second later)
Query 3: "restaurants in Goa" (1 second later)
Expected: All should complete successfully with proper memory context
```

---

## Test Execution Commands

### Basic Test
```bash
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "restaurants in Hyderabad", "user_id": "test_user_1"}'
```

### Location Switch Test
```bash
# Query 1
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "restaurants in Hyderabad", "user_id": "test_user_2"}'

# Query 2 - Different location
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "movie theaters in Goa", "user_id": "test_user_2"}'
```

### Memory Persistence Test
```bash
# Store preference
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "I love vegetarian food and budget options", "user_id": "test_user_3"}'

# Check if remembered
curl -X POST http://localhost:8001/memory-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "restaurants in Mumbai", "user_id": "test_user_3"}'
```

---

## Expected Verification Points

### ✅ Location Context
- [ ] Correct location extracted from query
- [ ] Location properly tagged in memory: `[Location] Query: ...`
- [ ] AI focuses on current query location, not historical
- [ ] Handles location switches correctly

### ✅ Memory Management
- [ ] Preferences stored and retrieved
- [ ] Memory context included in subsequent queries
- [ ] Different users have isolated memory
- [ ] Memory updates reflected in workflow_logs

### ✅ Semantic Caching
- [ ] Similar queries use cached responses
- [ ] Cache hit reduces latency significantly
- [ ] Different users don't share cache
- [ ] Similarity threshold (0.85) working correctly

### ✅ Model Comparison
- [ ] Both Flash and Pro responses generated
- [ ] Latency difference visible (Flash faster)
- [ ] Faster model correctly identified
- [ ] Both responses relevant and accurate

### ✅ Workflow Logs
- [ ] All expected fields present in JSON response
- [ ] has_memory_context accurate
- [ ] Destinations extracted correctly
- [ ] Fingerprint generated for caching

---

## Priority Test Order

1. **Critical**: Scenario 1.2 - Location Switch (most important bug fix)
2. **High**: Scenario 2.1 - Preference Learning (core memory feature)
3. **High**: Scenario 3.1 - Semantic Caching (performance optimization)
4. **Medium**: Scenario 5.1 - Follow-up Questions (user experience)
5. **Medium**: Scenario 8.1/8.2 - Workflow Logs (output verification)
6. **Low**: Scenario 10.1 - Concurrent Requests (scalability)
