# RuralHealthAssistant

# For Loading the model
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "/kaggle/input/tinyllama-final"  # your LoRA fine-tuned model path

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model).to("cuda")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload().to("cuda")
model.eval()

# For RAG module -
create a file Rag_module.py
