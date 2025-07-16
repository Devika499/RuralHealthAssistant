from fastapi import FastAPI, HTTPException, Depends, status, Body, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import logging
from typing import Optional, List
import os
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from peft import PeftModel
from sqlalchemy.orm import Session
from database import get_db, User as DBUser, ChatMessage as DBChatMessage, MedicalRecord as DBMedicalRecord, MedicationReminder as DBMedicationReminder
from rag_module import rag
from rlhf_module import rlhf
from indictrans2.translation_module import to_en, to_native
from symptom_checker import start_session as sc_start_session, answer_question as sc_answer_question, finish_session as sc_finish_session
import requests
import shutil
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Rural Healthcare AI Assistant...")
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")
    
    # Load RLHF reward model
    rlhf_success = rlhf.load_reward_model()
    if not rlhf_success:
        logger.warning("Failed to load RLHF reward model - continuing without RLHF")
    else:
        logger.info("RLHF reward model loaded successfully")
    
    yield
    # Shutdown
    logger.info("Shutting down Rural Healthcare AI Assistant...")

# Initialize FastAPI app
app = FastAPI(
    title="Rural Healthcare AI Assistant",
    description="AI-powered healthcare assistant for rural communities using TinyLlama",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    status: str = "success"
    rlhf_score: Optional[dict] = None  # RLHF scoring information

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    rlhf_loaded: bool = False
    rlhf_device: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserRegister(BaseModel):
    name: str
    mobileNumber: str  # Frontend sends mobileNumber
    password: str
    confirmPassword: Optional[str] = None  # Frontend sends this but we don't use it
    birthdate: Optional[str] = None
    preferredLanguage: Optional[str] = "hi"  # Frontend sends preferredLanguage
    village: Optional[str] = None
    age: Optional[int] = None
    
    class Config:
        # Allow field aliases for backward compatibility
        alias_generator = lambda string: string.replace("mobileNumber", "mobile_number").replace("preferredLanguage", "preferred_language")
        populate_by_name = True

class User(BaseModel):
    id: str
    name: str
    mobile_number: str
    village: Optional[str] = None
    age: Optional[int] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class MedicalHistoryRecord(BaseModel):
    id: Optional[str] = None
    user_id: str
    condition: str
    symptoms: str
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    date: str
    notes: Optional[str] = None
    report_file: Optional[str] = None

class ChatMessage(BaseModel):
    text: str
    sender: str
    timestamp: str
    language: str = "hi"

# Pydantic models for medication reminders
class MedicationReminderCreate(BaseModel):
    medication_name: str
    frequency: str  # daily, weekly, monthly
    time: str       # "09:00"
    start_date: str # ISO format
    end_date: Optional[str] = None

class MedicationReminderOut(BaseModel):
    id: str
    medication_name: str
    frequency: str
    time: str
    start_date: str
    end_date: Optional[str]
    created_at: str

# Pydantic models for symptom checker
class SymptomStartRequest(BaseModel):
    symptom: str
    language: str = 'en'

class SymptomStartResponse(BaseModel):
    session_id: str
    questions: list[str]

class SymptomAnswerRequest(BaseModel):
    session_id: str
    answer: str

class SymptomAnswerResponse(BaseModel):
    next_question: str = None
    done: bool
    current_q: int

class SymptomFinishRequest(BaseModel):
    session_id: str

class SymptomFinishResponse(BaseModel):
    advice: str

# Pydantic models for simplify-term endpoint
class SimplifyTermRequest(BaseModel):
    term: str
    language: str = 'en'

class SimplifyTermResponse(BaseModel):
    simplified: str

# Pydantic models for diet recommendation
class DietRecommendationRequest(BaseModel):
    prompt: str
    language: str

class DietRecommendationResponse(BaseModel):
    recommendation: str

# Authentication helper
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    # In production, validate JWT token here
    # For now, we'll use a simple token lookup where token is the user ID
    user = db.query(DBUser).filter(DBUser.id == token).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def classify_input(text):
    """Classify user input into different intents"""
    text_lower = text.lower().strip()

    # Check for greeting
    if any(greet in text_lower for greet in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        return "greeting"

    # Check for Q&A type
    if text_lower.startswith(("what is", "what are", "how does", "how to", "can i", "is it", "why", "which", "when")):
        return "qna"

    # Check for simplification (contains medical jargon or explicit request to simplify)
    if any(term in text_lower for term in ["explain", "simplify", "in simple words", "meaning of", "i don't understand"]):
        return "simplify"

    # Otherwise treat as symptom
    return "symptom"


def build_prompt(text, intent):
    """Build appropriate prompt based on intent"""
    if intent == "greeting":
        return "<|user|>: Hi! <|assistant|>: Hello! How can I help you with your health today?"

    if intent == "qna":
        return f"<|user|>: {text} <|assistant|>:"

    elif intent == "symptom":
        return f"<|user|>: I am experiencing the following symptoms: {text}. What could it be and what should I do? <|assistant|>:"

    elif intent == "simplify":
        return f"<|user|>: Please explain this in simple words for a rural person: {text} <|assistant|>:"

    # Default fallback
    return f"<|user|>: {text} <|assistant|>:"

def load_model():
    """Load TinyLlama model and tokenizer with 4-bit quantization"""
    global model, tokenizer
    
    try:
        logger.info("Loading TinyLlama model and tokenizer with 4-bit quantization...")
        
        # Check if CUDA is available for GPU quantization
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            logger.info("CUDA not available, using CPU with reduced precision")
            # For CPU, use regular loading with reduced precision
            bnb_config = None
        
        # Model path - adjust this path to where you store the TinyLlama model
        model_path = "models/tinyllama-final"  # Update this path as needed
        
        # Check if model exists locally, otherwise download
        if not os.path.exists(model_path):
            logger.info("Model not found locally, downloading from HuggingFace...")
            model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if this is a LoRA adapter model
        if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
            logger.info("Detected LoRA adapter model, loading base model and adapter...")
            # Load base model first
            base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            if bnb_config is not None:
                # Use quantization if available
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            else:
                # Use regular loading for CPU
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            # Load and apply LoRA adapter
            model = PeftModel.from_pretrained(model, model_path)
            logger.info("LoRA adapter loaded successfully")
        else:
            # Load regular model
            if bnb_config is not None:
                # Use quantization if available
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    quantization_config=bnb_config,
                    trust_remote_code=True
                )
            else:
                # Use regular loading for CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set model to evaluation mode
        model.eval()
        
        quantization_info = "with 4-bit quantization" if bnb_config is not None else "with CPU optimization"
        logger.info(f"Model loaded successfully on device: {model.device} {quantization_info}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Try fallback to regular loading without quantization
        try:
            logger.info("Attempting fallback to regular model loading...")
            model_path = "models/tinyllama-final"
            if not os.path.exists(model_path):
                model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
                base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
                model = PeftModel.from_pretrained(model, model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model.eval()
            logger.info(f"Model loaded successfully with fallback method on device: {model.device}")
            return True
            
        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed: {str(fallback_error)}")
            return False

def chat_with_model(user_input: str) -> tuple[str, str]:
    """Generate response using the loaded model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Classify input intent
        intent = classify_input(user_input)
        
        # Use RAG for Q&A intent
        if intent == "qna":
            response = rag.rag_infer(user_input, model, tokenizer)
            return response.strip(), intent
        
        # Build appropriate prompt for other intents
        prompt = build_prompt(user_input, intent)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("<|assistant|>:")[-1].strip()
        
        return response, intent
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Rural Healthcare AI Assistant API",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    device = str(next(model.parameters()).device) if model else "none"
    rlhf_status = rlhf.get_model_status()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device,
        rlhf_loaded=rlhf_status["is_loaded"],
        rlhf_device=rlhf_status["device"]
    )

# Authentication endpoints
@app.post("/token", response_model=TokenResponse)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """User login endpoint"""
    # Find user by mobile number
    user = db.query(DBUser).filter(DBUser.mobile_number == user_data.username).first()
    
    if user and user.password == user_data.password:
        # In production, use proper password hashing
        return TokenResponse(access_token=user.id)
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/users/me", response_model=User)
async def get_current_user_info(current_user: DBUser = Depends(get_current_user)):
    """Get current user information"""
    # Convert SQLAlchemy object to Pydantic model
    return User(
        id=str(current_user.id),
        name=str(current_user.name),
        mobile_number=str(current_user.mobile_number),
        village=current_user.village,
        age=current_user.age
    )

@app.post("/debug/register")
async def debug_register(request: dict):
    """Debug endpoint to see what data is being sent"""
    return {
        "received_data": request,
        "message": "Debug endpoint - check the received data format"
    }

@app.post("/users/", response_model=User)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """User registration endpoint"""
    # Check if user already exists
    existing_user = db.query(DBUser).filter(DBUser.mobile_number == user_data.mobileNumber).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Calculate age from birthdate if provided
    age = None
    if user_data.birthdate:
        try:
            birth_date = datetime.strptime(user_data.birthdate, "%Y-%m-%d")
            age = (datetime.now() - birth_date).days // 365
        except:
            age = None
    
    # Create new user
    user_id = str(uuid.uuid4())
    db_user = DBUser(
        id=user_id,
        name=user_data.name,
        mobile_number=user_data.mobileNumber,
        password=user_data.password,  # In production, hash the password
        village=user_data.village,
        age=age,
        preferred_language=user_data.preferredLanguage
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return User(
        id=user_id,
        name=user_data.name,
        mobile_number=user_data.mobileNumber,
        village=user_data.village,
        age=age
    )

@app.patch("/users/{user_id}", response_model=User)
async def update_user(user_id: str, updates: dict, current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update user information"""
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update user data
    for key, value in updates.items():
        if hasattr(current_user, key) and key in ["name", "village", "age"]:
            setattr(current_user, key, value)
    
    db.commit()
    db.refresh(current_user)
    
    return User(
        id=current_user.id,
        name=current_user.name,
        mobile_number=current_user.mobile_number,
        village=current_user.village,
        age=current_user.age
    )

# Chat endpoints
@app.get("/chat/history")
async def get_chat_history(current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get chat history for current user"""
    messages = db.query(DBChatMessage).filter(DBChatMessage.user_id == current_user.id).order_by(DBChatMessage.timestamp).all()
    
    return [
        {
            "text": msg.text,
            "sender": msg.sender,
            "timestamp": msg.timestamp.isoformat(),
            "language": msg.language
        }
        for msg in messages
    ]

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Main chat endpoint with translation support"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # 1. Translate user message to English and detect source language
        english_prompt, src_lang = to_en(request.message)
        
        # 2. Generate response in English using TinyLlama
        response_en, intent = chat_with_model(english_prompt)
        
        # 3. Translate response back to user's language (skip if English)
        if src_lang == "en":
            response_native = response_en
        else:
            response_native = to_native(response_en, tgt_iso=src_lang)
        
        # 4. Score response using RLHF (if available)
        rlhf_score = None
        if rlhf.is_loaded:
            rlhf_score = rlhf.score_response(request.message, response_en)
            logger.info(f"RLHF Score: {rlhf_score}")
        
        # 5. Store user message
        user_message = DBChatMessage(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            text=request.message,
            sender="user",
            language=src_lang,
            intent=intent
        )
        db.add(user_message)
        
        # 6. Store bot response
        bot_message = DBChatMessage(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            text=response_native,
            sender="bot",
            language=src_lang,
            intent=intent
        )
        db.add(bot_message)
        
        db.commit()
        
        logger.info(f"Chat request processed - Intent: {intent}, User: {current_user.id}, Lang: {src_lang}")
        
        return ChatResponse(
            response=response_native,
            intent=intent,
            rlhf_score=rlhf_score
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Medical history endpoints
@app.get("/medical-history")
async def get_medical_history(current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get medical history for current user"""
    records = db.query(DBMedicalRecord).filter(DBMedicalRecord.user_id == current_user.id).order_by(DBMedicalRecord.date.desc()).all()
    
    return [
        {
            "id": record.id,
            "user_id": record.user_id,
            "condition": record.condition,
            "symptoms": record.symptoms,
            "diagnosis": record.diagnosis,
            "treatment": record.treatment,
            "date": record.date.isoformat(),
            "notes": record.notes,
            "report_file": record.report_file
        }
        for record in records
    ]

@app.post("/medical-history", response_model=MedicalHistoryRecord)
async def create_medical_record(
    condition: str = Form(...),
    date: str = Form(...),
    notes: str = Form(""),
    user_id: str = Form(...),
    file: UploadFile = File(None),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new medical history record with optional file upload"""
    file_path = None
    if file:
        upload_dir = "uploads/medical_reports"
        os.makedirs(upload_dir, exist_ok=True)
        file_ext = os.path.splitext(file.filename)[1]
        unique_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(upload_dir, unique_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Accept both 'yyyy-mm-dd' and ISO format for date
    try:
        if 'T' in date:
            parsed_date = datetime.fromisoformat(date)
        else:
            parsed_date = datetime.fromisoformat(date + 'T00:00:00')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use yyyy-mm-dd or ISO format.")

    db_record = DBMedicalRecord(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        condition=condition,
        symptoms="",  # You can extend the form to accept symptoms
        diagnosis=None,
        treatment=None,
        date=parsed_date,
        notes=notes,
        report_file=file_path
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return MedicalHistoryRecord(
        id=db_record.id,
        user_id=db_record.user_id,
        condition=db_record.condition,
        symptoms=db_record.symptoms,
        diagnosis=db_record.diagnosis,
        treatment=db_record.treatment,
        date=db_record.date.isoformat(),
        notes=db_record.notes,
        report_file=db_record.report_file
    )

@app.put("/medical-history/{record_id}", response_model=MedicalHistoryRecord)
async def update_medical_record(record_id: str, record: MedicalHistoryRecord, current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update medical history record"""
    db_record = db.query(DBMedicalRecord).filter(
        DBMedicalRecord.id == record_id,
        DBMedicalRecord.user_id == current_user.id
    ).first()
    
    if not db_record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Update fields
    db_record.condition = record.condition
    db_record.symptoms = record.symptoms
    db_record.diagnosis = record.diagnosis
    db_record.treatment = record.treatment
    db_record.date = datetime.fromisoformat(record.date)
    db_record.notes = record.notes
    
    db.commit()
    db.refresh(db_record)
    
    return MedicalHistoryRecord(
        id=db_record.id,
        user_id=db_record.user_id,
        condition=db_record.condition,
        symptoms=db_record.symptoms,
        diagnosis=db_record.diagnosis,
        treatment=db_record.treatment,
        date=db_record.date.isoformat(),
        notes=db_record.notes
    )

@app.delete("/medical-history/{record_id}")
async def delete_medical_record(record_id: str, current_user: DBUser = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete medical history record"""
    db_record = db.query(DBMedicalRecord).filter(
        DBMedicalRecord.id == record_id,
        DBMedicalRecord.user_id == current_user.id
    ).first()
    
    if not db_record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    db.delete(db_record)
    db.commit()
    
    return {"message": "Record deleted successfully"}

# RLHF endpoints
@app.get("/rlhf/status")
async def get_rlhf_status():
    """Get RLHF reward model status"""
    return rlhf.get_model_status()

@app.post("/rlhf/reload")
async def reload_rlhf_model():
    """Reload the RLHF reward model"""
    try:
        success = rlhf.load_reward_model()
        if success:
            return {"message": "RLHF reward model reloaded successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload RLHF reward model")
    except Exception as e:
        logger.error(f"Error reloading RLHF reward model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reloading RLHF reward model: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload the model (useful for updates)"""
    try:
        success = load_model()
        if success:
            return {"message": "Model reloaded successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")

@app.post("/save-chat-to-history")
async def save_chat_to_history(
    chat_messages: list = Body(...),
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save the current chat as a medical record for the user and clear chat history."""
    if not chat_messages:
        raise HTTPException(status_code=400, detail="No chat messages provided")
    chat_text = "\n".join([
        f"{msg.get('sender', '')}: {msg.get('text', '')}" for msg in chat_messages
    ])
    record = DBMedicalRecord(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        condition="Chat Session",
        symptoms=chat_text,
        diagnosis=None,
        treatment=None,
        date=datetime.now(),
        notes="Saved from chat session"
    )
    db.add(record)
    # Delete all chat messages for this user
    db.query(DBChatMessage).filter(DBChatMessage.user_id == current_user.id).delete()
    db.commit()
    return {"message": "Chat saved to medical history and chat history cleared"}

# Medication Reminder Endpoints
@app.post("/medications/", response_model=MedicationReminderOut)
async def add_medication_reminder(
    reminder: MedicationReminderCreate,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    reminder_id = str(uuid.uuid4())
    start_date = datetime.fromisoformat(reminder.start_date)
    end_date = datetime.fromisoformat(reminder.end_date) if reminder.end_date else None
    db_reminder = DBMedicationReminder(
        id=reminder_id,
        user_id=current_user.id,
        medication_name=reminder.medication_name,
        frequency=reminder.frequency,
        time=reminder.time,
        start_date=start_date,
        end_date=end_date,
        created_at=datetime.utcnow()
    )
    db.add(db_reminder)
    db.commit()
    db.refresh(db_reminder)
    return MedicationReminderOut(
        id=db_reminder.id,
        medication_name=db_reminder.medication_name,
        frequency=db_reminder.frequency,
        time=db_reminder.time,
        start_date=db_reminder.start_date.isoformat(),
        end_date=db_reminder.end_date.isoformat() if db_reminder.end_date else None,
        created_at=db_reminder.created_at.isoformat()
    )

@app.get("/medications/", response_model=List[MedicationReminderOut])
async def list_medication_reminders(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    reminders = db.query(DBMedicationReminder).filter(DBMedicationReminder.user_id == current_user.id).all()
    return [
        MedicationReminderOut(
            id=r.id,
            medication_name=r.medication_name,
            frequency=r.frequency,
            time=r.time,
            start_date=r.start_date.isoformat(),
            end_date=r.end_date.isoformat() if r.end_date else None,
            created_at=r.created_at.isoformat()
        ) for r in reminders
    ]

@app.delete("/medications/{reminder_id}")
async def delete_medication_reminder(
    reminder_id: str,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    reminder = db.query(DBMedicationReminder).filter(DBMedicationReminder.id == reminder_id, DBMedicationReminder.user_id == current_user.id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    db.delete(reminder)
    db.commit()
    return {"detail": "Reminder deleted"}

# Symptom Checker Endpoints
@app.post("/symptom/start", response_model=SymptomStartResponse)
async def symptom_start(
    req: SymptomStartRequest,
    current_user: DBUser = Depends(get_current_user)
):
    session_id, questions = sc_start_session(current_user.id, req.symptom, req.language)
    return SymptomStartResponse(session_id=session_id, questions=questions)

@app.post("/symptom/answer", response_model=SymptomAnswerResponse)
async def symptom_answer(
    req: SymptomAnswerRequest,
    current_user: DBUser = Depends(get_current_user)
):
    next_q, current_q, done = sc_answer_question(req.session_id, req.answer)
    return SymptomAnswerResponse(next_question=next_q, done=done, current_q=current_q)

@app.post("/symptom/finish", response_model=SymptomFinishResponse)
async def symptom_finish(
    req: SymptomFinishRequest,
    current_user: DBUser = Depends(get_current_user)
):
    advice = sc_finish_session(req.session_id, model, tokenizer)
    return SymptomFinishResponse(advice=advice)

@app.post("/simplify-term", response_model=SimplifyTermResponse)
async def simplify_term(
    req: SimplifyTermRequest,
    current_user: DBUser = Depends(get_current_user)
):
    # Translate term to English if needed
    if req.language != 'en':
        term_en, _ = to_en(req.term)
    else:
        term_en = req.term
    # Build prompt for TinyLlama
    prompt = f"<|user|>: Please explain the medical term '{term_en}' in simple words for a rural person. <|assistant|>:"
    # Generate explanation
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if '<|assistant|>:' in decoded:
        explanation_en = decoded.split('<|assistant|>:')[-1].strip()
    else:
        explanation_en = decoded.strip()
    # Translate explanation to requested language if needed
    if req.language != 'en':
        explanation_native = to_native(explanation_en, tgt_iso=req.language)
    else:
        explanation_native = explanation_en
    return SimplifyTermResponse(simplified=explanation_native)

@app.post("/diet-recommendation/", response_model=DietRecommendationResponse)
async def diet_recommendation(
    req: DietRecommendationRequest,
    current_user: DBUser = Depends(get_current_user)
):
    """Get a diet recommendation from Groq Llama API, with translation support."""
    try:
        # 1. Translate user prompt to English
        prompt_en, src_lang = to_en(req.prompt)

        # 2. Call Groq Llama API
        llama_api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-70b-8192",  # or another Llama model as supported by Groq
            "messages": [
                {"role": "system", "content": "You are a helpful healthcare assistant. Provide practical, culturally appropriate diet recommendations based on the user's request."},
                {"role": "user", "content": prompt_en}
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        response = requests.post(llama_api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        llama_response_en = response.json()["choices"][0]["message"]["content"].strip()

        # 3. Translate response back to user's language
        if req.language == "en":
            recommendation = llama_response_en
        else:
            recommendation = to_native(llama_response_en, tgt_iso=req.language)

        return DietRecommendationResponse(recommendation=recommendation)
    except Exception as e:
        logger.error(f"Diet recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get diet recommendation.")

# Mount static files for medical reports
app.mount("/uploads/medical_reports", StaticFiles(directory="uploads/medical_reports"), name="medical_reports")

if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    if load_model():
        logger.info("Model loaded successfully, starting server...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Set to True for development
            log_level="info"
        )
    else:
        logger.error("Failed to load model, exiting...")
