"""
LLM Client - Tập trung TẤT CẢ functions liên quan đến model
Các file khác chỉ cần import và gọi hàm từ đây.
"""

import gc
from typing import Optional
from config import API_KEYS, JUDGE_API, JUDGE_MODEL

# ============================================================
# MODEL CACHE
# ============================================================

_loaded_models = {}  # {model_name: (model, tokenizer)}
_current_key_index = {"groq": 0, "gemini": 0, "openai": 0}

# ============================================================
# PUBLIC API - Các file khác dùng những hàm này
# ============================================================

def call_llm(prompt: str, model: str, max_tokens: int = 512) -> str:
    """
    Gọi LLM với tên model.
    Tự động detect: HuggingFace (có /) hoặc API model.
    
    Args:
        prompt: Prompt gửi đến LLM
        model: Tên model (vd: "Qwen/Qwen2.5-7B-Instruct" hoặc "llama-3.1-8b-instant")
        max_tokens: Số tokens tối đa
    
    Returns:
        Response text từ LLM
    """
    if "/" in model:  # HuggingFace model
        return _call_huggingface(prompt, model, max_tokens)
    else:  # API model
        return _call_api(prompt, model, max_tokens)


def call_judge(prompt: str) -> str:
    """Gọi LLM-as-Judge với config mặc định."""
    return _call_api(prompt, JUDGE_MODEL, 1000)


def unload_model():
    """Unload tất cả models và giải phóng GPU memory."""
    global _loaded_models
    _loaded_models.clear()
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    
    print("🗑️ Model unloaded, GPU cleared")


def is_model_loaded(model: str) -> bool:
    """Kiểm tra model đã load chưa."""
    return model in _loaded_models


# ============================================================
# INTERNAL - HuggingFace
# ============================================================

def _call_huggingface(prompt: str, model: str, max_tokens: int) -> str:
    """Gọi HuggingFace model với 4-bit quantization và chat template đúng cách."""
    global _loaded_models
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        # Load model nếu chưa có
        if model not in _loaded_models:
            print(f"📥 Loading: {model} (4-bit)...")
            
            # 4-bit quantization config - tăng tốc 2-3x
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model)
            
            # Set pad_token nếu chưa có
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            hf_model = AutoModelForCausalLM.from_pretrained(
                model,
                quantization_config=quantization_config,
                device_map="auto"
            )
            _loaded_models[model] = (hf_model, tokenizer)
            print(f"✅ Loaded!")
        
        hf_model, tokenizer = _loaded_models[model]
        
        # Tạo messages format cho chat model
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template nếu model hỗ trợ
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # Fallback nếu model không hỗ trợ chat template
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(hf_model.device)
        input_length = inputs["input_ids"].shape[1]
        
        # Generate với stop conditions tốt hơn
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 256),
            do_sample=False,  # Dùng greedy decoding để output ổn định hơn
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Giảm lặp
        )
        
        # Chỉ lấy phần response mới (bỏ prompt)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        return f"[ERROR] {e}"


# ============================================================
# INTERNAL - APIs (Groq, Gemini, OpenAI)
# ============================================================

def _get_key(api: str) -> str:
    """Lấy API key hiện tại."""
    keys = API_KEYS.get(api, [])
    if not keys:
        return ""
    idx = _current_key_index.get(api, 0) % len(keys)
    return keys[idx]


def _call_api(prompt: str, model: str, max_tokens: int) -> str:
    """Gọi API model - tự detect API từ model name."""
    if "llama" in model.lower() or "gemma" in model.lower() or "mixtral" in model.lower():
        return _call_groq(prompt, model, max_tokens)
    elif "gemini" in model.lower():
        return _call_gemini(prompt, model, max_tokens)
    elif "gpt" in model.lower():
        return _call_openai(prompt, model, max_tokens)
    else:
        return _call_groq(prompt, model, max_tokens)


def _call_groq(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=_get_key("groq"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _call_gemini(prompt: str, model: str, max_tokens: int) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_get_key("gemini"))
        llm = genai.GenerativeModel(model)
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _call_openai(prompt: str, model: str, max_tokens: int) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_get_key("openai"))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"
