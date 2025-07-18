# Rural Healthcare AI Assistant - Backend

A FastAPI-based backend service that provides an AI-powered healthcare assistant for rural communities using the TinyLlama model with RLHF (Reinforcement Learning from Human Feedback) integration.

## Features

- **TinyLlama Model Integration**: Uses TinyLlama-1.1B-Chat-v1.0 for natural language processing
- **RLHF Integration**: Reward model for response quality assessment and improvement
- **Intent Classification**: Automatically classifies user inputs into Q&A, symptom description, or simplification requests
- **Contextual Prompts**: Builds appropriate prompts based on user intent
- **RESTful API**: Clean API endpoints for chat functionality
- **Health Monitoring**: Built-in health check endpoints
- **CORS Support**: Configured for frontend integration
- **PostgreSQL Integration**: Persistent storage for users, chat history, and medical records

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Model Setup

The application will automatically download the TinyLlama model from HuggingFace on first run. Alternatively, you can:

1. Download the model manually to `models/tinyllama/` directory
2. Update the `model_path` in `main.py` to point to your local model

### 3. RLHF Reward Model Setup

The RLHF module requires a trained reward model:

1. Place your trained reward model in `models/reward_model/` directory
2. The model should include:
   - `adapter_model.safetensors` - The trained model weights
   - `tokenizer.json` - Tokenizer configuration
   - `adapter_config.json` - Model configuration
   - Other necessary model files

### 4. Database Setup

Follow the instructions in `SETUP_DATABASE.md` to set up PostgreSQL database.

### 5. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- **GET** `/health` - Check server, model, and RLHF status

### Chat
- **POST** `/chat` - Main chat endpoint
  - Request body: `{"message": "your message", "user_id": "optional_user_id"}`
  - Response: `{"response": "ai_response", "intent": "qna|symptom|simplify", "status": "success", "rlhf_score": {...}}`

### Authentication
- **POST** `/token` - User login
- **POST** `/users/` - User registration
- **GET** `/users/me` - Get current user info

### Chat History
- **GET** `/chat/history` - Get user's chat history

### Medical History
- **GET** `/medical-history` - Get user's medical records
- **POST** `/medical-history` - Create new medical record
- **PUT** `/medical-history/{record_id}` - Update medical record
- **DELETE** `/medical-history/{record_id}` - Delete medical record

### Model Management
- **POST** `/reload-model` - Reload the main model
- **POST** `/rlhf/reload` - Reload the RLHF reward model

### RLHF Management
- **GET** `/rlhf/status` - Get RLHF reward model status

## RLHF Integration

The system includes RLHF (Reinforcement Learning from Human Feedback) functionality:

### Reward Model
- **Purpose**: Evaluates response quality based on human feedback
- **Output**: Probability scores across 3 rank classes (0, 1, 2)
- **Integration**: Automatically scores each generated response

### Response Scoring
Each chat response includes RLHF scoring information:
```json
{
  "response": "AI generated response",
  "intent": "symptom",
  "status": "success",
  "rlhf_score": {
    "scores": [0.1, 0.3, 0.6],
    "predicted_rank": 2
  }
}
```

### Testing RLHF
Run the test script to verify RLHF functionality:
```bash
python test_rlhf.py
```

## Intent Classification

The system automatically classifies user inputs into three categories:

1. **Q&A** (`qna`): Questions starting with "what is", "how to", "why", etc.
2. **Symptom** (`symptom`): Default category for describing health symptoms
3. **Simplify** (`simplify`): Requests to explain medical terms in simple language

## Model Configuration

The TinyLlama model is configured with:
- **Max tokens**: 150
- **Temperature**: 0.8
- **Top-p**: 0.95
- **Repetition penalty**: 1.1
- **Device**: Auto-detection (CPU/GPU)
- **Quantization**: 4-bit quantization for memory efficiency

The RLHF reward model is configured with:
- **Num labels**: 3 (rank classes)
- **Max length**: 384 tokens
- **Device**: Auto-detection (CPU/GPU)

## Development

### Environment Variables
- Set `CUDA_VISIBLE_DEVICES` to control GPU usage
- Set `TRANSFORMERS_CACHE` to specify model cache directory

### Logging
The application uses Python's logging module with INFO level by default. Check console output for model loading and request processing logs.

## Troubleshooting

### Model Loading Issues
1. Ensure sufficient disk space for model download (~2GB)
2. Check internet connection for initial download
3. Verify CUDA installation if using GPU

### RLHF Issues
1. Verify reward model files are present in `models/reward_model/`
2. Check reward model configuration matches training setup
3. Run `python test_rlhf.py` to diagnose issues

### Memory Issues
1. Use CPU-only mode by setting `device_map="cpu"`
2. Reduce `max_new_tokens` in generation parameters
3. Consider using model quantization

### API Issues
1. Check CORS configuration for frontend integration
2. Verify request format matches Pydantic models
3. Check server logs for detailed error messages

## Production Deployment

For production deployment:

1. Update CORS origins in `main.py`
2. Set up proper logging configuration
3. Use a production WSGI server like Gunicorn
4. Implement rate limiting and authentication
5. Set up monitoring and health checks
6. Configure database connection pooling
7. Set up backup and recovery procedures

## License

This project is part of the Rural Healthcare AI Assistant system. 