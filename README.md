# 🏥 Rural Healthcare AI Assistant

A **FastAPI-based backend service** that powers an AI-driven healthcare assistant for **rural communities**.  
It uses the **TinyLlama-1.1B-Chat-v1.0** model for NLP and integrates **RLHF (Reinforcement Learning from Human Feedback)** to improve response quality.

---

## ✨ Functionalities
- Q&A (`qna`)
- Symptom Description (`symptom`)
- Simplification Requests (`simplify`)
- Diet Recommendations (`diet`)
- Medication Reminders (`medication`)
- Medical Document Storage (`med_doc`)

---
## 🌐 Language Support
- Supports **22 Indic languages** using **IndicTrans** for translation.  
- Enables multilingual conversations for rural healthcare access.  

---

## ⚙️ Model Setup
- By default, the app auto-downloads **TinyLlama** from Hugging Face on first run.  
- To use a local model:  
  1. Download the model into `models/tinyllama/`  
  2. Update `model_path` in `main.py` with your local path  

---

## 🤖 RLHF Reward Model Setup
- Place trained reward model files inside `models/reward_model/`:
  - `adapter_model.safetensors`  
  - `tokenizer.json`  
  - `adapter_config.json`  
  **Example Score Block**
```json
"rlhf_score": {
  "scores": [0.10, 0.30, 0.60],
  "predicted_rank": 2
}
```
---
## 🔑 API Key Setup
- In **`main.py`**, replace `GroqAPIkey` with your own API key.  
- This is required for **Diet Recommendations** functionality.  

---

## 🗄️ Database Setup
- Follow `SETUP_DATABASE.md` to configure **PostgreSQL**.

---

## 🚀 Running the Server
```bash
python main.py
```

---
## 📌 Important Notes

- Ensure at least 2GB free disk space for model download.
- Internet connection required for first model download.
- Supports both CPU and GPU (auto-detect).
- Quantization (4-bit) enabled for memory efficiency.
- RLHF integration provides quality scoring for AI responses.
- Multilingual support makes it accessible to rural communities.
