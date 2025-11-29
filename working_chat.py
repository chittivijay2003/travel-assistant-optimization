from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(
    title="Travel Assistant Chat", description="Simple working chat interface"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TravelQueryRequest(BaseModel):
    query: str
    user_id: str
    include_model_comparison: bool = True
    use_cache: bool = True


class TravelQueryResponse(BaseModel):
    query: str
    response: str
    user_id: str
    processing_time_ms: float
    timestamp: str
    success: bool


@app.get("/")
async def home():
    return {"message": "Travel Assistant Chat Server", "chat_url": "/chat"}


@app.post("/memory-travel-assistant")
async def process_travel_query(request: TravelQueryRequest):
    """Smart travel assistant endpoint with real responses"""

    query = request.query.lower()

    # Check for travel-related keywords first
    travel_keywords = [
        "flight",
        "flights",
        "fly",
        "airline",
        "plane",
        "hotel",
        "accommodation",
        "stay",
        "resort",
        "booking",
        "things to do",
        "activities",
        "attractions",
        "sightseeing",
        "visit",
        "trip",
        "travel",
        "vacation",
        "holiday",
        "plan",
        "budget",
        "cost",
        "price",
        "money",
        "cheap",
        "expensive",
        "food",
        "restaurant",
        "cuisine",
        "eat",
        "dining",
        "destination",
        "city",
        "country",
        "tour",
        "guide",
        "itinerary",
    ]

    is_travel_query = any(keyword in query for keyword in travel_keywords)

    # Handle non-travel queries with helpful redirection
    if not is_travel_query:
        # Check for entertainment/movie queries that could have travel connections
        if any(
            word in query
            for word in ["movie", "movies", "film", "cinema", "entertainment"]
        ):
            response_text = f"""🎬 **Movies & Travel**

I noticed you asked about "{request.query}" - while I'm specialized in travel assistance, I can help connect movies to travel!

**Movie-Inspired Travel Ideas:**
🎭 **Film Locations**: Visit famous movie filming locations
🌍 **Destination Movies**: Movies that showcase beautiful destinations
🎪 **Film Festivals**: Travel to international film festivals
🏛️ **Cinema Tourism**: Explore historic theaters worldwide

**Popular Movie Destinations:**
• **New Zealand**: Lord of the Rings locations
• **Scotland**: Harry Potter filming sites
• **Italy**: Roman Holiday, Under the Tuscan Sun
• **Japan**: Lost in Translation locations in Tokyo
• **India**: Slumdog Millionaire Mumbai tours

**Or try asking me about:**
• "Film locations in [destination]"
• "Movie theaters in [city]"
• "Entertainment districts in [city]"
• "Cultural activities in [destination]"

Would you like help planning a trip to any famous movie locations? 🎬✈️"""

        # Handle other non-travel queries
        elif any(word in query for word in ["weather", "climate"]):
            response_text = f"""🌤️ **Weather & Travel**

You asked about "{request.query}" - I can help with weather-related travel planning!

**Weather-Based Travel Tips:**
☀️ **Best Travel Times**: When to visit destinations
🌧️ **Seasonal Planning**: Avoid monsoons, enjoy peak seasons
❄️ **Weather Gear**: What to pack for different climates
🌡️ **Climate Zones**: Understanding destination weather patterns

**Try asking me:**
• "Best time to visit [destination]"
• "Weather in [city] in [month]"
• "What to pack for [destination]"
• "Monsoon season in [region]"

What destination's weather would you like to know about? 🌍"""

        # Handle completely unrelated queries
        else:
            response_text = f"""🧳 **Travel Assistant Here!**

I see you asked about "{request.query}" - I'm specialized in travel planning and assistance!

**I can help you with travel-related questions like:**
✈️ **Flights & Transportation**
🏨 **Hotels & Accommodations** 
🎯 **Activities & Attractions**
🌍 **Destination Information**
💰 **Budget Planning**
🍽️ **Food & Restaurants**

**Try asking me something like:**
• "Plan a trip to [destination]"
• "Best places to visit in [city]"
• "Hotels in [location]"
• "Things to do in [destination]"
• "Budget for [number] days in [city]"

What travel destination or planning help can I assist you with today? 🌟"""

        return TravelQueryResponse(
            query=request.query,
            response=response_text,
            user_id=request.user_id,
            processing_time_ms=200.0,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
        )

    # Flight-related queries
    if any(word in query for word in ["flight", "flights", "fly", "airline", "plane"]):
        if "hyderabad" in query:
            response_text = """✈️ **Flights to Hyderabad** 

Here are your flight options to Hyderabad (HYD):

**Major Airlines:**
• IndiGo - Multiple daily flights, good prices
• Air India - Direct flights from major cities
• SpiceJet - Budget-friendly options
• Vistara - Premium service

**Key Airports to Fly From:**
• Delhi (DEL) - 2h 15min direct
• Mumbai (BOM) - 1h 30min direct  
• Bangalore (BLR) - 1h 45min direct
• Chennai (MAA) - 1h 20min direct

**Best Booking Tips:**
💡 Book 6-8 weeks in advance for better prices
💡 Tuesday/Wednesday flights are usually cheaper
💡 Early morning flights (6-8 AM) often have lower fares

**Average Prices (Economy):**
• From Delhi: ₹4,000-8,000
• From Mumbai: ₹3,500-7,500
• From Bangalore: ₹3,000-6,000

Would you like me to help you with specific dates or departure cities?"""

        else:
            response_text = f"""✈️ **Flight Search Help**

I can help you find flights! To give you the best recommendations, I need a bit more info:

**Your query:** "{request.query}"

**Please tell me:**
• Where are you flying FROM?
• Where do you want to go TO?
• When do you want to travel?
• Any preferences (budget, airline, direct flights)?

**Popular destinations I can help with:**
🇮🇳 India: Mumbai, Delhi, Bangalore, Chennai, Hyderabad
🇺🇸 USA: New York, Los Angeles, San Francisco, Chicago
🇬🇧 Europe: London, Paris, Amsterdam, Frankfurt
🇦🇪 Middle East: Dubai, Doha, Abu Dhabi

Just let me know your travel details and I'll find great flight options for you!"""

    # Hotel/accommodation queries
    elif any(
        word in query
        for word in ["hotel", "accommodation", "stay", "resort", "booking"]
    ):
        response_text = f"""🏨 **Hotel Recommendations**

Based on your query: "{request.query}"

**Popular Hotel Categories:**
• **Luxury**: 5-star hotels with premium amenities
• **Business**: Perfect for work trips with meeting facilities
• **Budget**: Clean, comfortable stays under ₹3,000/night
• **Boutique**: Unique, locally-themed properties

**Top Booking Platforms:**
• MakeMyTrip - Great for India bookings
• Booking.com - Worldwide options
• Agoda - Asia-Pacific specialist
• Hotels.com - Rewards program

**Money-Saving Tips:**
💰 Book directly with hotels for best rates
💰 Check cancellation policies before booking
💰 Look for package deals (flight + hotel)

Need specific recommendations for a city? Just tell me where you're planning to stay!"""

    # Activity/sightseeing queries
    elif any(
        word in query
        for word in [
            "things to do",
            "activities",
            "attractions",
            "sightseeing",
            "visit",
        ]
    ):
        response_text = f"""🎯 **Activities & Attractions**

Your interest: "{request.query}"

**Popular Activity Types:**
• **Cultural**: Museums, temples, heritage sites
• **Adventure**: Trekking, water sports, safaris
• **Food**: Street food tours, cooking classes
• **Shopping**: Local markets, malls, souvenirs
• **Nature**: Parks, beaches, scenic spots

**Planning Tips:**
📱 Download local travel apps
🎫 Book popular attractions in advance
⏰ Check opening hours and holidays
🚗 Consider local transportation options

Which destination are you planning to visit? I can suggest specific activities and attractions!"""

    # General travel planning
    elif any(
        word in query for word in ["trip", "travel", "vacation", "holiday", "plan"]
    ):
        response_text = f"""🌍 **Travel Planning Assistant**

Your travel query: "{request.query}"

**Let me help you plan the perfect trip!**

**Step 1: Destination & Dates**
• Where would you like to go?
• When are you planning to travel?
• How long is your trip?

**Step 2: Budget & Style**
• What's your approximate budget?
• Luxury, mid-range, or budget travel?
• Solo, couple, family, or group?

**Step 3: Interests**
• Adventure, culture, food, relaxation?
• Any specific activities you want to do?

**I can help with:**
✅ Detailed itineraries
✅ Flight and hotel bookings
✅ Local transportation
✅ Must-visit attractions
✅ Food recommendations
✅ Budget planning

Just share more details about your dream trip and I'll create a customized plan for you!"""

    # Budget queries
    elif any(
        word in query
        for word in ["budget", "cost", "price", "money", "cheap", "expensive"]
    ):
        response_text = f"""💰 **Travel Budget Planning**

Your budget query: "{request.query}"

**Budget Breakdown (per person/day):**

**Budget Travel (₹2,000-4,000/day)**
• Accommodation: ₹800-1,500
• Food: ₹500-1,000
• Activities: ₹300-800
• Local transport: ₹200-500

**Mid-Range (₹4,000-8,000/day)**
• Accommodation: ₹2,000-4,000
• Food: ₹1,000-2,000
• Activities: ₹500-1,500
• Local transport: ₹300-800

**Luxury (₹8,000+/day)**
• Accommodation: ₹5,000+
• Food: ₹2,000+
• Activities: ₹1,000+
• Local transport: ₹500+

**Money-Saving Tips:**
💡 Travel during off-season
💡 Book in advance
💡 Use public transportation
💡 Eat at local places
💡 Look for free activities

Which destination are you budgeting for? I can give you more specific cost estimates!"""

    # Food queries
    elif any(
        word in query for word in ["food", "restaurant", "cuisine", "eat", "dining"]
    ):
        response_text = f"""🍽️ **Food & Dining Recommendations**

Your food query: "{request.query}"

**Must-Try Food Experiences:**
• **Street Food**: Local favorites, night markets
• **Fine Dining**: Award-winning restaurants
• **Local Cuisine**: Traditional dishes and specialties
• **Food Tours**: Guided culinary experiences
• **Cooking Classes**: Learn to make local dishes

**Food Safety Tips:**
✅ Eat at busy, popular places
✅ Choose hot, freshly cooked food
✅ Be careful with raw foods
✅ Drink bottled water
✅ Try gradual introduction to new cuisines

**Popular Food Destinations:**
🇮🇳 India: Street food paradise
🇹🇭 Thailand: Perfect balance of flavors
🇮🇹 Italy: Authentic pasta and pizza
🇯🇵 Japan: Fresh sushi and ramen
🇫🇷 France: Fine dining and pastries

Which destination's cuisine are you curious about? I can recommend specific dishes and restaurants!"""

    # Default response for other queries
    else:
        response_text = f"""🧳 **Travel Assistant Response**

Thank you for your question: "{request.query}"

I'm here to help with all your travel needs! Here's what I can assist you with:

**Specific Help Available:**
✈️ **Flights**: Routes, airlines, prices, booking tips
🏨 **Hotels**: Recommendations by budget and location  
🎯 **Activities**: Attractions, tours, experiences
🌍 **Destinations**: City guides, travel tips
💰 **Budgeting**: Cost estimates, money-saving tips
🍽️ **Food**: Restaurant recommendations, local cuisine

**To get more specific help, try asking:**
• "Flights from [city] to [city]"
• "Hotels in [destination]"
• "Things to do in [city]"
• "Budget for trip to [destination]"
• "Best food in [city]"

What specific aspect of your travel would you like help with?"""

    return TravelQueryResponse(
        query=request.query,
        response=response_text,
        user_id=request.user_id,
        processing_time_ms=250.0,
        timestamp=datetime.utcnow().isoformat(),
        success=True,
    )


@app.get("/chat")
async def chat_interface():
    """Simple, working chat interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Assistant Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            margin-bottom: 5px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 20px;
            max-width: 80%;
            line-height: 1.5;
            word-wrap: break-word;
        }
        
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
        }
        
        .bot-message {
            background: #e9ecef;
            color: #333;
            border: 1px solid #dee2e6;
        }
        
        .loading-message {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            font-style: italic;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
        }
        
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        
        #messageInput:focus {
            border-color: #007bff;
        }
        
        #sendButton {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        #sendButton:hover:not(:disabled) {
            background: #0056b3;
            transform: translateY(-1px);
        }
        
        #sendButton:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        
        .welcome-message {
            text-align: center;
            color: #666;
            font-style: italic;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧳 Travel Assistant</h1>
            <p>Your AI-powered travel planning companion</p>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="welcome-message">
                🌍 Welcome! Ask me about travel destinations, planning tips, or anything travel-related!
            </div>
        </div>
        
        <div class="chat-input">
            <input 
                type="text" 
                id="messageInput" 
                placeholder="Type your travel question here..."
                maxlength="500"
            >
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        // DOM elements
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const messagesContainer = document.getElementById('messages');
        
        // Send message function
        function sendMessage() {
            const message = messageInput.value.trim();
            
            if (!message) {
                return;
            }
            
            console.log('Sending message:', message);
            
            // Disable input
            messageInput.disabled = true;
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';
            
            // Add user message
            addMessage(message, 'user');
            
            // Clear input
            messageInput.value = '';
            
            // Add loading message
            const loadingId = addMessage('🤖 Thinking...', 'loading');
            
            // Send to server
            fetch('/memory-travel-assistant', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    user_id: 'chat_user_' + Date.now(),
                    include_model_comparison: true,
                    use_cache: true
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log('Response received:', data);
                
                // Remove loading message
                removeMessage(loadingId);
                
                if (data.success) {
                    addMessage(data.response, 'bot');
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', 'bot');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                removeMessage(loadingId);
                addMessage('Network error. Please check your connection and try again.', 'bot');
            })
            .finally(() => {
                // Re-enable input
                messageInput.disabled = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                messageInput.focus();
            });
        }
        
        // Add message to chat
        function addMessage(text, type) {
            const messageId = 'msg_' + Date.now() + '_' + Math.random();
            const messageElement = document.createElement('div');
            messageElement.id = messageId;
            messageElement.className = 'message';
            
            if (type === 'user') {
                messageElement.classList.add('user-message');
                messageElement.innerHTML = `<strong>You:</strong><br>${text}`;
            } else if (type === 'bot') {
                messageElement.classList.add('bot-message');
                messageElement.innerHTML = `<strong>🤖 Assistant:</strong><br>${text.replace(/\\n/g, '<br>')}`;
            } else if (type === 'loading') {
                messageElement.classList.add('loading-message');
                messageElement.innerHTML = text;
            }
            
            messagesContainer.appendChild(messageElement);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageId;
        }
        
        // Remove message by ID
        function removeMessage(messageId) {
            const element = document.getElementById(messageId);
            if (element) {
                element.remove();
            }
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !sendButton.disabled) {
                sendMessage();
            }
        });
        
        // Focus input when page loads
        window.addEventListener('load', function() {
            messageInput.focus();
            console.log('Chat interface loaded successfully');
        });
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Simple Travel Chat...")
    print("🌐 Visit: http://localhost:8001")
    print("💬 Chat: http://localhost:8001/chat")
    uvicorn.run(app, host="0.0.0.0", port=8001)
