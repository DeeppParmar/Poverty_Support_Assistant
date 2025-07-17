from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Poverty Alleviation AI Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API - Move API key to environment variable for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDCRCGTPZxyzRMAoFFlk7QkkIC5pk6FTmo")

# Global variable to store model
model = None

def initialize_gemini():
    """Initialize Gemini model with error handling"""
    global model
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Updated model name
        logger.info("✅ Gemini API configured successfully")
        
        # Test the model with a simple prompt
        test_response = model.generate_content("Say hello")
        logger.info(f"✅ Model test successful: {test_response.text[:50]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Error configuring Gemini API: {e}")
        model = None
        return False

# Initialize Gemini on startup
initialize_gemini()

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are a compassionate Poverty Alleviation Assistant helping low-income individuals and families access support services. Your role is to:

1. Provide information about welfare schemes, government benefits, and social programs
2. Help with job search resources, training programs, and employment opportunities
3. Guide users to food assistance programs, food banks, and nutrition support
4. Share information about healthcare access, free clinics, and medical assistance
5. Offer financial support resources, emergency aid, and budgeting advice
6. Provide clear, empathetic responses in simple language
7. Always maintain a supportive, non-judgmental tone
8. Include practical next steps and contact information when possible

Remember: You're helping people in difficult situations. Be kind, patient, and focus on actionable solutions. If you don't know specific local information, guide them to appropriate resources or suggest they contact local social services.

Keep responses concise but helpful, around 2-3 paragraphs maximum."""

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <body>
        <h1>Error: index.html not found</h1>
        <p>Please make sure index.html is in the same directory as this Python file.</p>
        </body>
        </html>
        """, status_code=404)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(message: ChatMessage):
    """Handle chat messages and return AI responses"""
    try:
        logger.info(f"📨 Received message: {message.message}")
        
        # Check if model is available
        if not model:
            logger.error("❌ Model not initialized, attempting to reinitialize...")
            if not initialize_gemini():
                return ChatResponse(
                    response="⚠️ The AI assistant is not configured properly. Please contact the administrator. In the meantime, you can contact your local social services office or call 211 for immediate assistance.",
                    status="error"
                )
        
        # Validate input
        if not message.message or not message.message.strip():
            return ChatResponse(
                response="Please enter a message for me to help you with.",
                status="error"
            )
        
        # Create the full prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {message.message}\n\nPlease provide a helpful, compassionate response focused on practical assistance and resources."
        
        logger.info(f"🤖 Sending prompt to Gemini...")
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=1000,
            temperature=0.7,
        )
        
        # Generate response using Gemini with safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Check if response was generated
        if not response:
            raise Exception("No response generated from Gemini")
        
        # Check if response was blocked
        if response.prompt_feedback.block_reason:
            logger.error(f"❌ Response blocked: {response.prompt_feedback.block_reason}")
            raise Exception(f"Response blocked: {response.prompt_feedback.block_reason}")
        
        # Extract response text
        if not response.text or not response.text.strip():
            raise Exception("Empty response text from Gemini")
        
        response_text = response.text.strip()
        logger.info(f"✅ Successfully generated response: {response_text[:100]}...")
        
        return ChatResponse(
            response=response_text,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        
        # Provide helpful fallback response
        fallback_response = """I apologize, but I'm having trouble connecting right now. Here are some immediate resources you can contact:

📞 **Call 211** - Free 24/7 helpline for local resources
🏢 **Local Social Services Office** - Government assistance programs
🍽️ **Food Banks** - Search "food bank near me" for local food assistance
💼 **Job Centers** - Government employment services and job training

Please try asking your question again in a moment, or contact these resources directly for immediate help."""
        
        return ChatResponse(
            response=fallback_response,
            status="error"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Poverty Alleviation AI Assistant",
        "api_configured": model is not None,
        "gemini_api_key_present": bool(GEMINI_API_KEY)
    }

@app.get("/test")
async def test_gemini():
    """Test endpoint to check Gemini API connection"""
    try:
        if not model:
            return {"status": "error", "message": "Model not initialized"}
        
        test_response = model.generate_content("Hello, please respond with 'API is working correctly'")
        return {
            "status": "success", 
            "message": "Gemini API is working",
            "test_response": test_response.text if test_response else "No response"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Poverty Alleviation AI Assistant...")
    print(f"📊 API Key configured: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"🤖 Model initialized: {'Yes' if model else 'No'}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)