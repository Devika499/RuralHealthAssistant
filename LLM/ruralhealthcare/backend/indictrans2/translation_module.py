import os
import sys
import sentencepiece
from IndicTransToolkit.processor import IndicProcessor
import torch
from langdetect import detect  # fallback when script detection fails

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftConfig, LoraModel, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors

# Try to import IndicProcessor
try:
    from IndicTransToolkit.processor import IndicProcessor
except ModuleNotFoundError:
    print("[ERROR] IndicTransToolkit not found. Install with:\n"
          "pip install --use-pep517 git+https://github.com/VarunGumma/IndicTransToolkit.git")
    sys.exit(1)

# ───────────────────────── Configuration ──────────────────────────
# Update these paths if you have local models, else use HuggingFace names
N2E_MODEL = "ai4bharat/indictrans2-indic-en-1B"        # Indic → English
E2N_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M" # English → Indic
SRC_EN    = "eng_Latn"


# If you have local models, use:
# N2E_MODEL = "models/indictrans2-indic-en-1B"
# E2N_MODEL = "models/indictrans2-en-indic-dist-200M"

device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_FULL = {
    "te": "tel_Telu", "hi": "hin_Deva", "ta": "tam_Taml", "kn": "kan_Knda",
    "bn": "ben_Beng", "mr": "mar_Deva", "pa": "pan_Guru", "gu": "guj_Gujr",
    "ml": "mal_Mlym", "or": "ory_Orya", "en": "eng_Latn",
}
SRC_EN = "eng_Latn"

# ───────────────────────── Load Translation Models ─────────────────────────
print("[INFO] Loading IndicTrans‑2 models…")
tok_n2e = AutoTokenizer.from_pretrained(N2E_MODEL, trust_remote_code=True)
mod_n2e = AutoModelForSeq2SeqLM.from_pretrained(N2E_MODEL, trust_remote_code=True).to(device)
tok_e2n = AutoTokenizer.from_pretrained(E2N_MODEL, trust_remote_code=True)
mod_e2n = AutoModelForSeq2SeqLM.from_pretrained(E2N_MODEL, trust_remote_code=True).to(device)
ip = IndicProcessor(inference=True)

# ───────────────────────── Script‑based Language ID ─────────────────────────
SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)], "pa": [(0x0A00, 0x0A7F)], "gu": [(0x0A80, 0x0AFF)],
    "or": [(0x0B00, 0x0B7F)], "ta": [(0x0B80, 0x0BFF)], "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)], "ml": [(0x0D00, 0x0D7F)], "bn": [(0x0980, 0x09FF)],
}

def detect_script_iso(text: str) -> str | None:
    counts = {k: 0 for k in SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for iso, ranges in SCRIPT_RANGES.items():
            if any(start <= cp <= end for start, end in ranges):
                counts[iso] += 1
                break
    iso_best = max(counts, key=counts.get)
    return iso_best if counts[iso_best] else None

# ───────────────────────── Translation Helpers ─────────────────────────────
def to_en(text: str):
    """Translate Indic → English. Returns (english, src_iso)."""
    iso = detect_script_iso(text) or (detect(text) if text else "hi")
    iso = iso if iso in LANG_FULL else "hi"
    src_tag = LANG_FULL[iso]
    batch = ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=SRC_EN)
    inputs = tok_n2e(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mod_n2e.generate(**inputs, max_length=512, num_beams=5)
    decoded = tok_n2e.batch_decode(outputs, skip_special_tokens=True)
    english = ip.postprocess_batch(decoded, lang=SRC_EN)[0]
    return english, iso

def to_native(text: str, tgt_iso: str):
    """Translate English → Indic."""
    tgt_iso = tgt_iso if tgt_iso in LANG_FULL else "hi"
    tgt_tag = LANG_FULL[tgt_iso]
    batch = ip.preprocess_batch([text], src_lang=SRC_EN, tgt_lang=tgt_tag)
    inputs = tok_e2n(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = mod_e2n.generate(**inputs, max_length=512, num_beams=5)
    decoded = tok_e2n.batch_decode(outputs, skip_special_tokens=True)
    native  = ip.postprocess_batch(decoded, lang=tgt_tag)[0]
    return native
