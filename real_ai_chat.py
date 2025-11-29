import os
import hashlib
import json
import redis
import mem0
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class that reads from environment variables"""

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 8002))
    AI_MODEL = os.getenv("AI_MODEL", "gemini-pro")
    APP_TITLE = os.getenv("APP_TITLE", "AI Travel Assistant")
    APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Real LLM-powered travel chat")

    @classmethod
    def validate_required_config(cls):
        """Validate that required configuration is present"""
        if not cls.GOOGLE_API_KEY:
            print("❌ No GOOGLE_API_KEY found in environment variables!")
            print("🔧 Please add your API key to the .env file:")
            print("   GOOGLE_API_KEY=your_api_key_here")
            print("📝 Get your key from: https://makersuite.google.com/app/apikey")
            return False
        return True


# Initialize configuration
config = Config()

app = FastAPI(title=config.APP_TITLE, description=config.APP_DESCRIPTION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure Google Gemini API
def setup_gemini():
    """Setup Gemini API with API key from configuration"""
    if not config.validate_required_config():
        return None

    genai.configure(api_key=config.GOOGLE_API_KEY)
    try:
        # List available models for debugging
        print("🔍 Available models:")
        available_models = []
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"  - {model.name}")
                available_models.append(model.name)

        # Check if the configured model is available
        model_name = config.AI_MODEL
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        if model_name not in available_models:
            print(f"❌ Model '{config.AI_MODEL}' not found!")
            print("💡 Available models:")
            for available in available_models[:5]:  # Show first 5
                clean_name = available.replace("models/", "")
                print(f"   - {clean_name}")
            print("💡 Update your .env file with one of these models")
            return None

        model = genai.GenerativeModel(config.AI_MODEL)
        print(f"✅ {config.AI_MODEL} initialized successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to initialize {config.AI_MODEL}: {e}")
        print("💡 Try using 'gemini-2.5-flash' or 'gemini-flash-latest' in .env")
        return None


# Initialize Gemini
gemini_model = setup_gemini()


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
    ai_powered: bool
    model_used: str


@app.get("/")
async def home():
    api_status = "✅ Connected" if gemini_model else "❌ No API Key"
    return {
        "message": "Real AI Travel Assistant",
        "chat_url": "/chat",
        "ai_status": api_status,
        "model": config.AI_MODEL if gemini_model else "No Model",
        "host": config.SERVER_HOST,
        "port": config.SERVER_PORT,
    }


@app.get("/status")
async def status():
    """Get detailed status information"""
    return {
        "ai_connected": bool(gemini_model),
        "model": config.AI_MODEL,
        "config": {
            "host": config.SERVER_HOST,
            "port": config.SERVER_PORT,
            "app_title": config.APP_TITLE,
            "has_api_key": bool(config.GOOGLE_API_KEY),
        },
    }


@app.post("/memory-travel-assistant")
async def process_travel_query(request: TravelQueryRequest):
    """Real AI-powered travel assistant"""

    start_time = datetime.now()

    if gemini_model:
        try:
            # Create a travel-focused prompt
            prompt = f"""You are an expert AI Travel Assistant. Help users with their travel-related questions.
            
User Question: {request.query}

Please provide helpful, specific, and actionable travel advice. Include:
- Specific recommendations when possible
- Practical tips and insights
- Cost estimates if relevant
- Safety considerations if applicable
- Local cultural insights when appropriate

If the question isn't travel-related, politely redirect them to travel topics while being helpful.

Keep your response conversational, informative, and engaging. Use emojis appropriately to make it friendly."""

            response = gemini_model.generate_content(prompt)
            ai_response = response.text

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return TravelQueryResponse(
                query=request.query,
                response=ai_response,
                user_id=request.user_id,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                ai_powered=True,
                model_used="Google Gemini Pro",
            )

        except Exception as e:
            error_msg = str(e)
            print(f"AI Error: {error_msg}")

            # Check for specific error types
            if "not found" in error_msg.lower() or "404" in error_msg:
                fallback_response = f"""🔧 **Model Configuration Issue**
                
Your question: "{request.query}"

The AI model '{config.AI_MODEL}' is not available. This usually means:
1. The model name is outdated
2. The API version changed  
3. The model is not supported in your region

**Quick Fix:** Update your .env file to use:
```
AI_MODEL=gemini-1.5-flash
```

**Current Error:** {error_msg}

Would you like me to help with basic travel information while you fix the AI configuration?"""

            elif (
                "api key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                fallback_response = f"""🔑 **API Key Issue**
                
Your question: "{request.query}"

There's an authentication problem with your Google API key:
**Error:** {error_msg}

**Solutions:**
1. Check if your API key is valid
2. Ensure you have Gemini API access enabled
3. Verify billing is set up (if required)
4. Get a new key from: https://makersuite.google.com/app/apikey

Meanwhile, I can provide basic travel guidance. What would you like to know?"""

            else:
                fallback_response = f"""🤖 **AI Temporarily Unavailable**
            
Your question: "{request.query}"

AI Error Details: {error_msg}

I'm having trouble connecting to the AI service right now. Here's what I can help you with:

✈️ **Travel Planning**: Destinations, itineraries, flights
🏨 **Accommodations**: Hotels, booking tips
🎯 **Activities**: Attractions, local experiences  
💰 **Budget Planning**: Cost estimates, saving tips
🍽️ **Food & Culture**: Local cuisine, restaurants

Please try again, or ask me something specific like:
• "Best time to visit Japan"
• "Budget hotels in Paris"  
• "Things to do in Bangkok"

What would you like to know about travel?"""

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return TravelQueryResponse(
                query=request.query,
                response=fallback_response,
                user_id=request.user_id,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow().isoformat(),
                success=True,
                ai_powered=False,
                model_used="Fallback Mode",
            )

    else:
        # No API key available
        no_api_response = f"""🔧 **Setup Required**
        
Your question: "{request.query}"

This travel assistant needs a Google API key to provide AI-powered responses.

**To enable real AI responses:**
1. Get a free API key from: https://makersuite.google.com/app/apikey
2. Set it in your environment: `export GOOGLE_API_KEY='your_key'`
3. Restart the server

**Without AI, I can still help with basic travel info:**
• Popular destinations and attractions
• General travel tips and advice
• Budget planning guidelines
• Booking recommendations

Would you like me to set up the AI for you, or do you have specific travel questions?"""

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TravelQueryResponse(
            query=request.query,
            response=no_api_response,
            user_id=request.user_id,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
            success=True,
            ai_powered=False,
            model_used="No API Key",
        )


@app.get("/chat")
async def chat_interface():
    """Real AI chat interface"""
    api_status = "🟢 AI Connected" if gemini_model else "🔴 API Key Needed"
    model_info = "Google Gemini Pro" if gemini_model else "Setup Required"

    return HTMLResponse(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real AI Travel Assistant</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        
        .chat-container {{
            width: 90%;
            max-width: 900px;
            height: 85vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .chat-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }}
        
        .chat-header h1 {{
            margin-bottom: 5px;
        }}
        
        .ai-status {{
            position: absolute;
            top: 15px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        
        .message {{
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 20px;
            max-width: 80%;
            line-height: 1.5;
            word-wrap: break-word;
        }}
        
        .user-message {{
            background: #007bff;
            color: white;
            margin-left: auto;
        }}
        
        .bot-message {{
            background: #e9ecef;
            color: #333;
            border: 1px solid #dee2e6;
            position: relative;
        }}
        
        .ai-badge {{
            position: absolute;
            top: -8px;
            right: 10px;
            background: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
            font-weight: bold;
        }}
        
        .loading-message {{
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            font-style: italic;
        }}
        
        .chat-input {{
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
        }}
        
        #messageInput {{
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }}
        
        #messageInput:focus {{
            border-color: #007bff;
        }}
        
        #sendButton {{
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        #sendButton:hover:not(:disabled) {{
            background: #0056b3;
            transform: translateY(-1px);
        }}
        
        #sendButton:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }}
        
        .welcome-message {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 15px;
            border-radius: 15px;
            border: 1px solid #dee2e6;
        }}
        
        .model-info {{
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="ai-status">{api_status}</div>
            <h1>🤖 Real AI Travel Assistant</h1>
            <p>Powered by artificial intelligence</p>
            <div class="model-info">{model_info}</div>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="welcome-message">
                🌍 Welcome to your AI-powered travel assistant!<br>
                Ask me anything about travel - I'm powered by real artificial intelligence!
                <br><br>
                <strong>Status:</strong> {api_status}<br>
                <strong>Model:</strong> {model_info}
            </div>
        </div>
        
        <div class="chat-input">
            <input 
                type="text" 
                id="messageInput" 
                placeholder="Ask your AI travel assistant anything..."
                maxlength="500"
            >
            <button id="sendButton">Send to AI</button>
        </div>
    </div>

    <script>
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const messagesContainer = document.getElementById('messages');
        
        function sendMessage() {{
            const message = messageInput.value.trim();
            
            if (!message) {{
                return;
            }}
            
            console.log('Sending to AI:', message);
            
            messageInput.disabled = true;
            sendButton.disabled = true;
            sendButton.textContent = 'AI Thinking...';
            
            addMessage(message, 'user');
            messageInput.value = '';
            
            const loadingId = addMessage('🤖 AI is processing your request...', 'loading');
            
            fetch('/memory-travel-assistant', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{
                    query: message,
                    user_id: 'ai_user_' + Date.now(),
                    include_model_comparison: true,
                    use_cache: true
                }})
            }})
            .then(response => {{
                if (!response.ok) {{
                    throw new Error('AI service error');
                }}
                return response.json();
            }})
            .then(data => {{
                console.log('AI Response:', data);
                
                removeMessage(loadingId);
                
                if (data.success) {{
                    addMessage(data.response, 'bot', data.ai_powered, data.model_used, data.processing_time_ms);
                }} else {{
                    addMessage('Sorry, AI encountered an error. Please try again.', 'bot');
                }}
            }})
            .catch(error => {{
                console.error('Error:', error);
                removeMessage(loadingId);
                addMessage('Network error. Please check your connection and try again.', 'bot');
            }})
            .finally(() => {{
                messageInput.disabled = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send to AI';
                messageInput.focus();
            }});
        }}
        
        function addMessage(text, type, aiPowered = false, model = '', processingTime = 0) {{
            const messageId = 'msg_' + Date.now() + '_' + Math.random();
            const messageElement = document.createElement('div');
            messageElement.id = messageId;
            messageElement.className = 'message';
            
            if (type === 'user') {{
                messageElement.classList.add('user-message');
                messageElement.innerHTML = `<strong>You:</strong><br>${{text}}`;
            }} else if (type === 'bot') {{
                messageElement.classList.add('bot-message');
                let badge = '';
                if (aiPowered) {{
                    badge = '<div class="ai-badge">AI</div>';
                }}
                const timeInfo = processingTime ? `<div style="font-size: 10px; color: #666; margin-top: 5px;">⚡ ${{Math.round(processingTime)}}ms • ${{model}}</div>` : '';
                messageElement.innerHTML = `${{badge}}<strong>🤖 AI Assistant:</strong><br>${{text.replace(/\\n/g, '<br>')}}${{timeInfo}}`;
            }} else if (type === 'loading') {{
                messageElement.classList.add('loading-message');
                messageElement.innerHTML = text;
            }}
            
            messagesContainer.appendChild(messageElement);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageId;
        }}
        
        function removeMessage(messageId) {{
            const element = document.getElementById(messageId);
            if (element) {{
                element.remove();
            }}
        }}
        
        sendButton.addEventListener('click', sendMessage);
        
        messageInput.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter' && !sendButton.disabled) {{
                sendMessage();
            }}
        }});
        
        window.addEventListener('load', function() {{
            messageInput.focus();
            console.log('Real AI Travel Assistant loaded');
        }});
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting {config.APP_TITLE}...")
    print(f"🌐 Visit: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"💬 Chat: http://{config.SERVER_HOST}:{config.SERVER_PORT}/chat")
    if gemini_model:
        print(f"✅ AI Status: Connected to {config.AI_MODEL}")
    else:
        print("⚠️  AI Status: No API key found")
        print("🔧 Setup: Add GOOGLE_API_KEY to .env file")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
