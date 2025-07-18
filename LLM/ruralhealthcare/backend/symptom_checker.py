import uuid
from datetime import datetime
from indictrans2.translation_module import to_en, to_native
from typing import Dict
import torch

# In-memory session store: {session_id: {user_id, language, symptom, questions, answers, current_q}}
session_store: Dict[str, dict] = {}

# Placeholder for LLM question generation (replace with TinyLlama call)
def generate_followup_questions(symptom_en):
    # In real use, call TinyLlama with prompt to generate 3 questions
    return [
        "How long have you had this symptom?",
        "Do you have any other symptoms?",
        "Have you taken any medication for this?"
    ]

def tinyllama_generate(prompt, model, tokenizer):
    # Generate response using the loaded model
    if model is None or tokenizer is None:
        return "Model not loaded."
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the assistant's answer
    if '<|assistant|>:' in decoded:
        return decoded.split('<|assistant|>:')[-1].strip()
    return decoded.strip()

# Use TinyLlama for advice generation with a rural-friendly, simple prompt and debug prints
def generate_advice(symptom, questions, answers, language, model, tokenizer):
    # Translate everything to English for the prompt
    if language != 'en':
        symptom_en, _ = to_en(symptom)
        questions_en = [to_en(q)[0] for q in questions]
        answers_en = [to_en(a)[0] for a in answers]
    else:
        symptom_en = symptom
        questions_en = questions
        answers_en = answers
    prompt = (
        f"<|user|>: I am a rural patient. I am experiencing the following symptoms: {symptom_en}.\n"
    )
    for i, (q, a) in enumerate(zip(questions_en, answers_en), 1):
        prompt += f"Q{i}: {q} A{i}: {a}\n"
    prompt += (
        "Please give a short, simple, and practical advice in easy language for a rural person. "
        "Avoid medical jargon. What could it be and what should I do?\n<|assistant|>:"
    )
    advice_en = tinyllama_generate(prompt, model, tokenizer)
    print("Advice EN:", advice_en)  # Debug
    advice_native = to_native(advice_en, tgt_iso=language) if language != 'en' else advice_en
    print("Advice Native:", advice_native)  # Debug
    print("Language:", language)  # Debug
    return advice_native

# Start a new symptom checker session
def start_session(user_id, symptom, language):
    if language != 'en':
        symptom_en, _ = to_en(symptom)
        questions_en = generate_followup_questions(symptom_en)
        questions_native = [to_native(q, tgt_iso=language) for q in questions_en]
    else:
        symptom_en = symptom
        questions_en = generate_followup_questions(symptom_en)
        questions_native = questions_en
    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        'user_id': user_id,
        'language': language,
        'symptom': symptom,
        'symptom_en': symptom_en,
        'questions_en': questions_en,
        'questions_native': questions_native,
        'answers': [],
        'answers_en': [],
        'current_q': 0,
        'created_at': datetime.utcnow().isoformat()
    }
    print("Session created:", session_store[session_id])  # Debug
    return session_id, questions_native

# Record an answer and get next question (if any)
def answer_question(session_id, answer):
    session = session_store.get(session_id)
    if not session:
        return None, None, True
    lang = session['language']
    if lang != 'en':
        answer_en, _ = to_en(answer)
    else:
        answer_en = answer
    session['answers'].append(answer)
    session['answers_en'].append(answer_en)
    session['current_q'] += 1
    if session['current_q'] < len(session['questions_native']):
        next_q = session['questions_native'][session['current_q']]
        return next_q, session['current_q'], False
    else:
        return None, session['current_q'], True

# Finish session and generate advice
def finish_session(session_id, model, tokenizer):
    session = session_store.get(session_id)
    if not session:
        return None
    # Use original symptom/questions/answers and language for translation
    advice_native = generate_advice(
        session['symptom'],
        session['questions_native'],
        session['answers'],
        session['language'],
        model,
        tokenizer
    )
    return advice_native 