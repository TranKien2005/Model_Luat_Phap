"""
Config - CHỈ chứa cấu hình, không có hàm
Sửa ở đây để thay đổi models, API keys.
"""

import os

# ============================================================
# BENCHMARK MODELS - 5 models HuggingFace để đánh giá
# ============================================================

BENCHMARK_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",           # 7B - Apache 2.0
    "deepseek-ai/deepseek-llm-7b-chat",   # 7B - MIT (không cần xác thực)
    "mistralai/Mistral-7B-Instruct-v0.3",  # 7B - Apache 2.0
    "HuggingFaceH4/zephyr-7b-beta",        # 7B - MIT (không cần xác thực)
    "openchat/openchat-3.5-0106",          # 7B - Apache 2.0 (không cần xác thực)
]

# ============================================================
# API KEYS - Cho LLM-as-Judge
# ============================================================

API_KEYS = {
    "groq": [
        k.strip() for k in os.environ.get("GROQ_API_KEY", "YOUR_GROQ_KEY").split(",") if k.strip()
    ],
    "gemini": [
        k.strip() for k in os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_KEY").split(",") if k.strip()
    ],
    "openai": [
        k.strip() for k in os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY").split(",") if k.strip()
    ],
}

# ============================================================
# JUDGE CONFIG
# ============================================================

JUDGE_API = "groq"
JUDGE_MODEL = "llama-3.1-8b-instant"

# ============================================================
# TASK CONFIG
# ============================================================

TASK_NAMES = [
    "general_issue_binary",
    "case_type_classification",
    "judgment_outcome_prediction",
    "numerical_constraint_check",
    "eligibility_logic_verification",
    "clause_type_identification",
    "legal_entity_extraction",
    "functional_sentence_labelling",
    "argument_consistency_check",
    "reasoning_method_detection",
    "vn_legal_mcq"
]

# ============================================================
# PATHS
# ============================================================

DATA_DIR = "./data"
OUTPUT_DIR = "./responses"
