"""
Reasoning Evaluator - Đánh giá reasoning bằng LLM-as-Judge
Sử dụng rubric 4 tiêu chí:
1. Cấu trúc logic (0-1)
2. Viện dẫn luật (0-1)
3. Áp dụng fact (0-1)
4. Kết luận phù hợp (0-1)
"""

import json
import re
from typing import Dict, Any

# Import unified LLM caller từ config
from config import call_llm, get_judge_config

REASONING_RUBRIC_PROMPT = """Bạn là một chuyên gia pháp lý Việt Nam. Hãy đánh giá chất lượng phần lập luận (reasoning) của một câu trả lời pháp lý.

## CÂU HỎI GỐC:
{question}

## ĐÁP ÁN CHUẨN:
- Answer: {ref_answer}
- Reasoning: {ref_reasoning}

## CÂU TRẢ LỜI CỦA MODEL:
- Answer: {model_answer}
- Reasoning: {model_reasoning}

## TIÊU CHÍ ĐÁNH GIÁ (mỗi tiêu chí 0 hoặc 1 điểm):

1. **Cấu trúc logic**: Có đầy đủ 3 phần (Luật → Fact → Kết luận) và trình tự logic đúng không?
2. **Viện dẫn luật**: Có trích dẫn điều luật cụ thể và phù hợp với vấn đề không?
3. **Áp dụng fact**: Có liên hệ và trích dẫn đúng chi tiết từ tình huống không?
4. **Kết luận phù hợp**: Kết luận có logic từ luật + fact và phù hợp với answer đã chọn không?

## YÊU CẦU:
Trả lời theo format JSON:
{{
    "logic_structure": 0 hoặc 1,
    "law_citation": 0 hoặc 1,
    "fact_application": 0 hoặc 1,
    "conclusion_consistency": 0 hoặc 1,
    "total_score": tổng điểm 0-4,
    "explanation": "Giải thích ngắn gọn"
}}"""


def extract_json_from_judge_response(response: str) -> dict:
    """Trích xuất JSON từ response của judge."""
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
    
    return {"error": "Cannot parse JSON", "total_score": 0}


def evaluate_reasoning_with_llm(
    question: str,
    ref_answer: str,
    ref_reasoning: str,
    model_answer: str,
    model_reasoning: str,
    judge_api: str = None,
    judge_model: str = None
) -> Dict[str, Any]:
    """
    Đánh giá reasoning bằng LLM-as-Judge.
    
    Args:
        question: Câu hỏi gốc
        ref_answer: Đáp án chuẩn
        ref_reasoning: Reasoning chuẩn
        model_answer: Đáp án của model
        model_reasoning: Reasoning của model
        judge_api: API để dùng (None = dùng config mặc định)
        judge_model: Model cụ thể (None = dùng config mặc định)
    
    Returns:
        Dict với các điểm thành phần và tổng điểm normalized (0-1)
    """
    # Lấy config mặc định nếu không chỉ định
    if judge_api is None or judge_model is None:
        default_api, default_model = get_judge_config()
        judge_api = judge_api or default_api
        judge_model = judge_model or default_model
    
    # Tạo prompt
    prompt = REASONING_RUBRIC_PROMPT.format(
        question=question,
        ref_answer=ref_answer,
        ref_reasoning=ref_reasoning,
        model_answer=model_answer,
        model_reasoning=model_reasoning
    )
    
    # Gọi LLM qua unified interface
    response = call_llm(prompt, api=judge_api, model=judge_model, max_tokens=500)
    
    # Parse response
    result = extract_json_from_judge_response(response)
    
    # Nếu có lỗi, trả về điểm 0
    if "error" in result:
        return {
            "reasoning_score": 0.0,
            "logic_structure": 0,
            "law_citation": 0,
            "fact_application": 0,
            "conclusion_consistency": 0,
            "explanation": result.get("error", "Unknown error"),
            "raw_response": response
        }
    
    # Tính điểm normalized (0-1)
    total = result.get("total_score", 0)
    normalized_score = total / 4.0  # Max 4 điểm
    
    return {
        "reasoning_score": normalized_score,
        "logic_structure": result.get("logic_structure", 0),
        "law_citation": result.get("law_citation", 0),
        "fact_application": result.get("fact_application", 0),
        "conclusion_consistency": result.get("conclusion_consistency", 0),
        "explanation": result.get("explanation", ""),
        "raw_response": response
    }


# ============ CONVENIENCE FUNCTION ============

def evaluate_rule_application_reasoning(
    sample: dict,
    model_response_parsed: dict,
    judge_api: str = "groq"
) -> float:
    """
    Hàm tiện ích để đánh giá reasoning cho các task rule-application.
    
    Returns:
        float: Điểm reasoning normalized (0.0 - 1.0)
    """
    question = sample.get("question", "")
    ref_answer = sample.get("answer", "")
    ref_reasoning = sample.get("reasoning", "")
    
    model_answer = model_response_parsed.get("answer", "")
    model_reasoning = model_response_parsed.get("reasoning", "")
    
    # Nếu không có reasoning từ model, return 0
    if not model_reasoning:
        return 0.0
    
    result = evaluate_reasoning_with_llm(
        question=question,
        ref_answer=ref_answer,
        ref_reasoning=ref_reasoning,
        model_answer=model_answer,
        model_reasoning=model_reasoning,
        judge_api=judge_api
    )
    
    return result.get("reasoning_score", 0.0)


# ============ ENTITY EXTRACTION EVALUATION ============

ENTITY_EVAL_PROMPT = """Bạn là chuyên gia pháp lý Việt Nam. Hãy đánh giá xem entity được trích xuất có đúng không.

## CÂU HỎI GỐC:
{question}

## ĐÁP ÁN CHUẨN (Entity đúng):
{ref_entity}

## CÂU TRẢ LỜI CỦA MODEL:
{model_entity}

## TIÊU CHÍ ĐÁNH GIÁ:
1. **Đúng hoàn toàn**: Entity của model hoàn toàn khớp với đáp án chuẩn → 1.0 điểm
2. **Đúng một phần**: Entity đúng về bản chất nhưng có thêm/thiếu một số chi tiết nhỏ → 0.5 điểm
3. **Sai**: Entity sai hoàn toàn hoặc không liên quan → 0.0 điểm

## YÊU CẦU:
Trả lời theo format JSON:
{{
    "score": 0.0 hoặc 0.5 hoặc 1.0,
    "is_correct": true hoặc false,
    "explanation": "Giải thích ngắn gọn"
}}"""


def evaluate_entity_with_llm(
    question: str,
    ref_entity: str,
    model_entity: str,
    judge_api: str = None,
    judge_model: str = None
) -> Dict[str, Any]:
    """
    Đánh giá entity extraction bằng LLM-as-Judge.
    
    Args:
        question: Câu hỏi gốc
        ref_entity: Entity chuẩn
        model_entity: Entity của model
        judge_api: API để dùng (None = dùng config mặc định)
        judge_model: Model cụ thể (None = dùng config mặc định)
    
    Returns:
        Dict với score (0.0, 0.5, 1.0) và is_correct
    """
    # Lấy config mặc định nếu không chỉ định
    if judge_api is None or judge_model is None:
        default_api, default_model = get_judge_config()
        judge_api = judge_api or default_api
        judge_model = judge_model or default_model
    
    # Tạo prompt
    prompt = ENTITY_EVAL_PROMPT.format(
        question=question,
        ref_entity=ref_entity,
        model_entity=model_entity
    )
    
    # Gọi LLM qua unified interface
    response = call_llm(prompt, api=judge_api, model=judge_model, max_tokens=300)
    
    # Parse response
    result = extract_json_from_judge_response(response)
    
    # Nếu có lỗi, trả về điểm 0
    if "error" in result:
        return {
            "entity_score": 0.0,
            "is_correct": False,
            "explanation": result.get("error", "Unknown error"),
            "raw_response": response
        }
    
    score = result.get("score", 0.0)
    is_correct = result.get("is_correct", score >= 0.5)
    
    return {
        "entity_score": score,
        "is_correct": is_correct,
        "explanation": result.get("explanation", ""),
        "raw_response": response
    }

