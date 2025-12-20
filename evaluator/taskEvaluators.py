"""
Task Evaluators - Hàm đánh giá riêng cho từng task

Mỗi task có:
- create_prompt(): Tạo prompt yêu cầu trả về JSON
- evaluate(): Đánh giá response, trả về dict với:
    - conclusion_correct: bool - Đáp án đúng/sai
    - conclusion_score: float - Điểm conclusion (0.0 - 1.0)
    - reasoning_score: float hoặc None - Điểm reasoning (0.0 - 1.0), None nếu không áp dụng
    - details: dict - Chi tiết đánh giá

Phân loại:
- Category issue-spotting, interpretation, rhetorical: chỉ có conclusion_score
- Category rule-application: có cả conclusion_score và reasoning_score
"""

import json
import re
from typing import Tuple, Optional, Dict, Any


def extract_json_from_response(response: str) -> dict:
    """Trích xuất JSON từ response của model."""
    # Thử tìm JSON trong markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Thử tìm JSON object trực tiếp
    json_match = re.search(r'\{[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {}


# ============================================================
# TASK 1: general_issue_binary
# Category: issue-spotting
# Đánh giá: Accuracy (Có/Không)
# ============================================================

def create_prompt_general_issue_binary(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam. 

HƯỚNG DẪN: {instruction}

CÂU HỎI: {question}

Trả lời theo format JSON:
{{"answer": "Có" hoặc "Không"}}"""


def evaluate_general_issue_binary(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: issue-spotting
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    # Check answer
    is_correct = ref_answer in model_answer or model_answer.startswith(ref_answer)
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,  # Không áp dụng cho task này
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# TASK 2: case_type_classification
# Category: issue-spotting
# Đánh giá: Accuracy (6 loại)
# ============================================================

def create_prompt_case_type_classification(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

TÓM TẮT BẢN ÁN: {question}

Trả lời theo format JSON:
{{"answer": "Tên nhóm"}}

Các nhóm: Dân sự, Hành chính, Hình sự, Kinh doanh thương mại, Lao động, Hôn nhân và gia đình"""


def evaluate_case_type_classification(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: issue-spotting
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    is_correct = ref_answer in model_answer
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# TASK 3: judgment_outcome_prediction
# Category: rule-application
# Đánh giá: Answer (A/B/C/D) + Reasoning (TODO: thảo luận)
# ============================================================

def create_prompt_judgment_outcome_prediction(instruction: str, question: str) -> str:
    return f"""Bạn là thẩm phán Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "A/B/C/D", "reasoning": "Giải thích theo format: Theo luật... mà... nên..."}}"""


def evaluate_judgment_outcome_prediction(
    reference: dict, 
    response: str,
    evaluate_reasoning: bool = True,
    judge_api: str = "groq"
) -> Dict[str, Any]:
    """
    Category: rule-application
    Returns: dict với conclusion_score và reasoning_score riêng
    
    Args:
        evaluate_reasoning: True để đánh giá reasoning bằng LLM
        judge_api: "groq" hoặc "gemini"
    """
    ref_answer = reference.get("answer", "").strip().upper()
    ref_reasoning = reference.get("reasoning", "")
    question = reference.get("question", "")
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", "").strip().upper()
    model_reasoning = parsed.get("reasoning", "")
    
    # Nếu không parse được, thử tìm A/B/C/D trong response
    if not model_answer:
        for letter in ["A", "B", "C", "D"]:
            if response.strip().upper().startswith(letter):
                model_answer = letter
                break
    
    answer_correct = ref_answer == model_answer
    
    # Đánh giá reasoning bằng LLM
    reasoning_score = None
    reasoning_details = {}
    
    if evaluate_reasoning and model_reasoning:
        try:
            from reasoningEvaluator import evaluate_reasoning_with_llm
            result = evaluate_reasoning_with_llm(
                question=question,
                ref_answer=ref_answer,
                ref_reasoning=ref_reasoning,
                model_answer=model_answer,
                model_reasoning=model_reasoning,
                judge_api=judge_api
            )
            reasoning_score = result.get("reasoning_score", 0.0)
            reasoning_details = {
                "logic_structure": result.get("logic_structure", 0),
                "law_citation": result.get("law_citation", 0),
                "fact_application": result.get("fact_application", 0),
                "conclusion_consistency": result.get("conclusion_consistency", 0),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            reasoning_score = None
            reasoning_details = {"error": str(e)}
    
    return {
        "conclusion_correct": answer_correct,
        "conclusion_score": 1.0 if answer_correct else 0.0,
        "reasoning_score": reasoning_score,
        "details": {
            "reference_answer": ref_answer,
            "model_answer": model_answer,
            "reference_reasoning": ref_reasoning,
            "model_reasoning": model_reasoning,
            "reasoning_evaluation": reasoning_details
        }
    }


# ============================================================
# TASK 4: numerical_constraint_check
# Category: rule-application
# Đánh giá: Answer (A/B/C/D) + Reasoning (TODO)
# ============================================================

def create_prompt_numerical_constraint_check(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "A/B/C/D", "reasoning": "Giải thích với phép tính cụ thể"}}"""


def evaluate_numerical_constraint_check(
    reference: dict, 
    response: str,
    evaluate_reasoning: bool = True,
    judge_api: str = "groq"
) -> Dict[str, Any]:
    """
    Category: rule-application
    Returns: dict với conclusion_score và reasoning_score riêng
    """
    ref_answer = reference.get("answer", "").strip().upper()
    ref_reasoning = reference.get("reasoning", "")
    question = reference.get("question", "")
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", "").strip().upper()
    model_reasoning = parsed.get("reasoning", "")
    
    if not model_answer:
        for letter in ["A", "B", "C", "D"]:
            if response.strip().upper().startswith(letter):
                model_answer = letter
                break
    
    answer_correct = ref_answer == model_answer
    
    # Đánh giá reasoning bằng LLM
    reasoning_score = None
    reasoning_details = {}
    
    if evaluate_reasoning and model_reasoning:
        try:
            from reasoningEvaluator import evaluate_reasoning_with_llm
            result = evaluate_reasoning_with_llm(
                question=question,
                ref_answer=ref_answer,
                ref_reasoning=ref_reasoning,
                model_answer=model_answer,
                model_reasoning=model_reasoning,
                judge_api=judge_api
            )
            reasoning_score = result.get("reasoning_score", 0.0)
            reasoning_details = {
                "logic_structure": result.get("logic_structure", 0),
                "law_citation": result.get("law_citation", 0),
                "fact_application": result.get("fact_application", 0),
                "conclusion_consistency": result.get("conclusion_consistency", 0),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            reasoning_score = None
            reasoning_details = {"error": str(e)}
    
    return {
        "conclusion_correct": answer_correct,
        "conclusion_score": 1.0 if answer_correct else 0.0,
        "reasoning_score": reasoning_score,
        "details": {
            "reference_answer": ref_answer,
            "model_answer": model_answer,
            "reference_reasoning": ref_reasoning,
            "model_reasoning": model_reasoning,
            "reasoning_evaluation": reasoning_details
        }
    }


# ============================================================
# TASK 5: eligibility_logic_verification
# Category: rule-application
# Đánh giá: Answer (Có/Không) + Reasoning (TODO)
# ============================================================

def create_prompt_eligibility_logic_verification(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

CÂU HỎI: {question}

Trả lời theo format JSON:
{{"answer": "Có/Không", "reasoning": "Giải thích logic điều kiện"}}"""


def evaluate_eligibility_logic_verification(
    reference: dict, 
    response: str,
    evaluate_reasoning: bool = True,
    judge_api: str = "groq"
) -> Dict[str, Any]:
    """
    Category: rule-application
    Returns: dict với conclusion_score và reasoning_score riêng
    """
    ref_answer = reference.get("answer", "").strip().lower()
    ref_reasoning = reference.get("reasoning", "")
    question = reference.get("question", "")
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", "").strip().lower()
    model_reasoning = parsed.get("reasoning", "")
    
    if not model_answer:
        if "có" in response.lower()[:20]:
            model_answer = "có"
        elif "không" in response.lower()[:20]:
            model_answer = "không"
    
    answer_correct = ref_answer == model_answer
    
    # Đánh giá reasoning bằng LLM
    reasoning_score = None
    reasoning_details = {}
    
    if evaluate_reasoning and model_reasoning:
        try:
            from reasoningEvaluator import evaluate_reasoning_with_llm
            result = evaluate_reasoning_with_llm(
                question=question,
                ref_answer=ref_answer,
                ref_reasoning=ref_reasoning,
                model_answer=model_answer,
                model_reasoning=model_reasoning,
                judge_api=judge_api
            )
            reasoning_score = result.get("reasoning_score", 0.0)
            reasoning_details = {
                "logic_structure": result.get("logic_structure", 0),
                "law_citation": result.get("law_citation", 0),
                "fact_application": result.get("fact_application", 0),
                "conclusion_consistency": result.get("conclusion_consistency", 0),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            reasoning_score = None
            reasoning_details = {"error": str(e)}
    
    return {
        "conclusion_correct": answer_correct,
        "conclusion_score": 1.0 if answer_correct else 0.0,
        "reasoning_score": reasoning_score,
        "details": {
            "reference_answer": ref_answer,
            "model_answer": model_answer,
            "reference_reasoning": ref_reasoning,
            "model_reasoning": model_reasoning,
            "reasoning_evaluation": reasoning_details
        }
    }


# ============================================================
# TASK 6: clause_type_identification
# Category: interpretation
# Đánh giá: Accuracy (7 loại)
# ============================================================

def create_prompt_clause_type_identification(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "Tên loại"}}

Các loại: Thông tin đương sự, Nội dung vụ án, Nhận định của Tòa, Quyết định, Hiệu lực"""


def evaluate_clause_type_identification(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: interpretation
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    is_correct = ref_answer in model_answer
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# TASK 7: legal_entity_extraction
# Category: interpretation
# Đánh giá: LLM-as-Judge (TODO: implement)
# ============================================================

def create_prompt_legal_entity_extraction(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "Tên/Danh tính được yêu cầu"}}"""


def evaluate_legal_entity_extraction(
    reference: dict, 
    response: str,
    use_llm: bool = True,
    judge_api: str = "groq"
) -> Dict[str, Any]:
    """
    Category: interpretation
    Đánh giá entity extraction bằng LLM hoặc substring match
    
    Args:
        use_llm: True để dùng LLM đánh giá, False để dùng substring match
        judge_api: "groq" hoặc "gemini"
    """
    ref_answer = reference.get("answer", "").strip()
    question = reference.get("question", "")
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip()
    
    # Đánh giá bằng LLM
    entity_score = None
    entity_details = {}
    
    if use_llm and model_answer:
        try:
            from reasoningEvaluator import evaluate_entity_with_llm
            result = evaluate_entity_with_llm(
                question=question,
                ref_entity=ref_answer,
                model_entity=model_answer,
                judge_api=judge_api
            )
            entity_score = result.get("entity_score", 0.0)
            is_correct = result.get("is_correct", False)
            entity_details = {
                "score": entity_score,
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            # Fallback to substring match
            is_correct = ref_answer.lower() in model_answer.lower() or model_answer.lower() in ref_answer.lower()
            entity_score = 1.0 if is_correct else 0.0
            entity_details = {"error": str(e)}
    else:
        # Substring match fallback
        is_correct = ref_answer.lower() in model_answer.lower() or model_answer.lower() in ref_answer.lower()
        entity_score = 1.0 if is_correct else 0.0
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": entity_score,
        "reasoning_score": None,  # Entity task không có reasoning
        "details": {
            "reference": ref_answer,
            "model": model_answer,
            "entity_evaluation": entity_details
        }
    }


# ============================================================
# TASK 8: functional_sentence_labelling
# Category: rhetorical-understanding
# Đánh giá: Accuracy ([FACT], [LAW], [JUDGMENT])
# ============================================================

def create_prompt_functional_sentence_labelling(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "[FACT]" hoặc "[LAW]" hoặc "[JUDGMENT]"}}"""


def evaluate_functional_sentence_labelling(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: rhetorical-understanding
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    is_correct = ref_answer in model_answer
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# TASK 9: argument_consistency_check
# Category: rhetorical-understanding
# Đánh giá: Accuracy (Nhất quán/Mâu thuẫn)
# ============================================================

def create_prompt_argument_consistency_check(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

{question}

Trả lời theo format JSON:
{{"answer": "Nhất quán" hoặc "Mâu thuẫn"}}"""


def evaluate_argument_consistency_check(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: rhetorical-understanding
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    if "nhất quán" in ref_answer:
        is_correct = "nhất quán" in model_answer and "mâu thuẫn" not in model_answer
    else:
        is_correct = "mâu thuẫn" in model_answer
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# TASK 10: reasoning_method_detection
# Category: rhetorical-understanding
# Đánh giá: Accuracy ([Textualism]/[Purposivism])
# ============================================================

def create_prompt_reasoning_method_detection(instruction: str, question: str) -> str:
    return f"""Bạn là chuyên gia pháp lý Việt Nam.

HƯỚNG DẪN: {instruction}

CÁCH TÒA LẬP LUẬN: {question}

Trả lời theo format JSON:
{{"answer": "[Textualism]" hoặc "[Purposivism]"}}"""


def evaluate_reasoning_method_detection(reference: dict, response: str) -> Dict[str, Any]:
    """
    Category: rhetorical-understanding
    Returns: dict với conclusion_score (reasoning_score = None)
    """
    ref_answer = reference.get("answer", "").strip().lower()
    
    parsed = extract_json_from_response(response)
    model_answer = parsed.get("answer", response).strip().lower()
    
    if "textualism" in ref_answer:
        is_correct = "textualism" in model_answer
    else:
        is_correct = "purposivism" in model_answer
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": 1.0 if is_correct else 0.0,
        "reasoning_score": None,
        "details": {
            "reference": ref_answer,
            "model": model_answer
        }
    }


# ============================================================
# REGISTRY - Map task_name -> (create_prompt, evaluate)
# ============================================================

TASK_EVALUATORS = {
    "general_issue_binary": (
        create_prompt_general_issue_binary,
        evaluate_general_issue_binary
    ),
    "case_type_classification": (
        create_prompt_case_type_classification,
        evaluate_case_type_classification
    ),
    "judgment_outcome_prediction": (
        create_prompt_judgment_outcome_prediction,
        evaluate_judgment_outcome_prediction
    ),
    "numerical_constraint_check": (
        create_prompt_numerical_constraint_check,
        evaluate_numerical_constraint_check
    ),
    "eligibility_logic_verification": (
        create_prompt_eligibility_logic_verification,
        evaluate_eligibility_logic_verification
    ),
    "clause_type_identification": (
        create_prompt_clause_type_identification,
        evaluate_clause_type_identification
    ),
    "legal_entity_extraction": (
        create_prompt_legal_entity_extraction,
        evaluate_legal_entity_extraction
    ),
    "functional_sentence_labelling": (
        create_prompt_functional_sentence_labelling,
        evaluate_functional_sentence_labelling
    ),
    "argument_consistency_check": (
        create_prompt_argument_consistency_check,
        evaluate_argument_consistency_check
    ),
    "reasoning_method_detection": (
        create_prompt_reasoning_method_detection,
        evaluate_reasoning_method_detection
    )
}


def get_evaluator(task_name: str):
    """Lấy hàm create_prompt và evaluate cho task."""
    return TASK_EVALUATORS.get(task_name, (None, None))
