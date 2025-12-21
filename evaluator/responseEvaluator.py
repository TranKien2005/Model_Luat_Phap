"""
Response Evaluator - Đánh giá responses đã lưu từ folder responses

Chức năng:
- Đọc responses từ folder đã generate
- Trích xuất JSON linh hoạt từ model_response
- Đánh giá bằng Accuracy và ROUGE-L

Usage:
    python responseEvaluator.py --responses-dir ../responses/kaggle/working/responses
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

# ============================================================
# TASK CATEGORIES
# ============================================================

# Task → Category mapping
# Rule tasks: dùng cho cả Rule Conclusion (accuracy) và Rule Application (reasoning)
# Interpretation: gồm clause_type (accuracy) + entity_extraction (ROUGE-L)

TASK_CATEGORIES = {
    # Issue-spotting: accuracy
    "general_issue_binary": "issue-spotting",
    "case_type_classification": "issue-spotting",
    
    # Rule tasks: dùng cho 2 categories
    # - Rule Conclusion: accuracy của answer
    # - Rule Application: ROUGE-L của reasoning
    "judgment_outcome_prediction": "rule",
    "numerical_constraint_check": "rule",
    "eligibility_logic_verification": "rule",
    
    # Interpretation: gộp cả 2 task
    "clause_type_identification": "interpretation",
    "legal_entity_extraction": "interpretation",
    
    # Rhetorical-understanding: accuracy
    "functional_sentence_labelling": "rhetorical-understanding",
    "argument_consistency_check": "rhetorical-understanding",
    "reasoning_method_detection": "rhetorical-understanding",
    
    # Rule-recall: kiến thức pháp luật MCQ (accuracy)
    "vn_legal_mcq": "rule-recall",
}

# 6 Categories cuối cùng
CATEGORY_NAMES = {
    "issue-spotting": "Issue Spotting",
    "rule-conclusion": "Rule Conclusion",
    "rule-application": "Rule Application",
    "interpretation": "Interpretation",
    "rhetorical-understanding": "Rhetorical Understanding",
    "rule-recall": "Rule Recall",
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def extract_json_from_response(response: str) -> dict:
    """
    Trích xuất JSON từ response - linh hoạt xử lý nhiều trường hợp lỗi.
    
    Returns:
        dict: JSON object hoặc {} nếu không tìm được
    """
    if not response or not response.strip():
        return {}
    
    # Pattern 1: JSON trong markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{[^`]*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Pattern 2: Tìm JSON object có key "answer"
    json_patterns = re.findall(r'\{[^{}]*"answer"[^{}]*\}', response)
    for json_str in json_patterns:
        try:
            return json.loads(json_str)
        except:
            continue
    
    # Pattern 3: JSON object đơn giản
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    return {}


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Tính ROUGE-L score (F1) giữa reference và hypothesis.
    
    Args:
        reference: Text chuẩn
        hypothesis: Text của model
    
    Returns:
        float: ROUGE-L F1 score (0.0 - 1.0)
    """
    if not reference or not hypothesis:
        return 0.0
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    # Tìm LCS length bằng SequenceMatcher
    matcher = SequenceMatcher(None, ref_tokens, hyp_tokens)
    lcs_length = sum(block.size for block in matcher.get_matching_blocks())
    
    # Tính precision, recall, F1
    precision = lcs_length / len(hyp_tokens) if hyp_tokens else 0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def check_accuracy(reference: str, model_answer: str, task_name: str) -> bool:
    """
    Kiểm tra accuracy cho các loại task khác nhau.
    
    Returns:
        bool: True nếu đúng, False nếu sai
    """
    if not model_answer:
        return False
    
    ref = reference.strip().lower()
    ans = model_answer.strip().lower()
    
    # Binary tasks: Có/Không
    if task_name in ["general_issue_binary", "eligibility_logic_verification"]:
        if ref == "có":
            return "có" in ans and "không" not in ans[:10]
        else:
            return "không" in ans
    
    # Multiple choice: A/B/C/D
    if task_name in ["judgment_outcome_prediction", "numerical_constraint_check"]:
        return ans.startswith(ref) or ref == ans
    
    # Classification tasks
    if task_name in ["case_type_classification", "clause_type_identification", 
                     "functional_sentence_labelling"]:
        return ref in ans
    
    # Consistency check
    if task_name == "argument_consistency_check":
        if "nhất quán" in ref:
            return "nhất quán" in ans and "mâu thuẫn" not in ans
        else:
            return "mâu thuẫn" in ans
    
    # Reasoning method
    if task_name == "reasoning_method_detection":
        if "textualism" in ref:
            return "textualism" in ans
        else:
            return "purposivism" in ans
    
    # Entity extraction - substring match
    if task_name == "legal_entity_extraction":
        return ref in ans or ans in ref or ref == ans
    
    return ref == ans


# ============================================================
# EVALUATOR FUNCTIONS
# ============================================================

def evaluate_response(
    task_name: str,
    reference: dict,
    model_response: str
) -> Dict[str, Any]:
    """
    Đánh giá một response.
    
    Returns:
        dict với:
        - conclusion_correct: bool
        - conclusion_score: float (0 hoặc 1)
        - reasoning_score: float (ROUGE-L) hoặc None
        - parsed_answer: str
    """
    # Trích xuất JSON từ response
    parsed = extract_json_from_response(model_response)
    model_answer = parsed.get("answer", "")
    model_reasoning = parsed.get("reasoning", "")
    
    # Lấy reference answer
    ref_answer = reference.get("answer", "")
    ref_reasoning = reference.get("reasoning", "")
    
    # Tính accuracy
    is_correct = check_accuracy(ref_answer, model_answer, task_name)
    conclusion_score = 1.0 if is_correct else 0.0
    
    # Tính ROUGE-L cho reasoning (nếu có)
    reasoning_score = None
    if ref_reasoning and model_reasoning:
        reasoning_score = compute_rouge_l(ref_reasoning, model_reasoning)
    
    # Tính ROUGE-L cho entity extraction
    entity_score = None
    if task_name == "legal_entity_extraction" and model_answer:
        entity_score = compute_rouge_l(ref_answer, model_answer)
    
    return {
        "conclusion_correct": is_correct,
        "conclusion_score": conclusion_score,
        "reasoning_score": reasoning_score,
        "entity_score": entity_score,
        "parsed_answer": model_answer,
        "parsed_reasoning": model_reasoning,
        "json_extracted": bool(parsed)
    }


def evaluate_task_file(task_file: Path) -> Dict[str, Any]:
    """
    Đánh giá toàn bộ responses trong một file task.
    
    Returns:
        dict với summary và detailed results
    """
    with open(task_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task_name = data.get("task_name", "")
    model_name = data.get("model_name", "")
    responses = data.get("responses", [])
    
    results = []
    total_correct = 0
    total_reasoning_score = 0.0
    total_entity_score = 0.0
    reasoning_count = 0
    entity_count = 0
    json_failed = 0
    
    for item in responses:
        reference = item.get("reference", {})
        model_response = item.get("model_response", "")
        
        eval_result = evaluate_response(task_name, reference, model_response)
        
        if eval_result["conclusion_correct"]:
            total_correct += 1
        
        if not eval_result["json_extracted"]:
            json_failed += 1
        
        if eval_result["reasoning_score"] is not None:
            total_reasoning_score += eval_result["reasoning_score"]
            reasoning_count += 1
        
        if eval_result["entity_score"] is not None:
            total_entity_score += eval_result["entity_score"]
            entity_count += 1
        
        results.append({
            "id": item.get("id"),
            "question": item.get("question", "")[:100] + "...",
            "ref_answer": reference.get("answer", ""),
            "model_answer": eval_result["parsed_answer"],
            "correct": eval_result["conclusion_correct"],
            "reasoning_score": eval_result["reasoning_score"],
            "entity_score": eval_result["entity_score"]
        })
    
    # Tính summary
    total = len(responses)
    accuracy = total_correct / total if total > 0 else 0
    avg_reasoning = total_reasoning_score / reasoning_count if reasoning_count > 0 else None
    avg_entity = total_entity_score / entity_count if entity_count > 0 else None
    
    return {
        "task_name": task_name,
        "model_name": model_name,
        "total_samples": total,
        "correct": total_correct,
        "accuracy": accuracy,
        "avg_reasoning_rouge_l": avg_reasoning,
        "avg_entity_rouge_l": avg_entity,
        "json_extraction_failed": json_failed,
        "timestamp": datetime.now().isoformat(),
        "details": results
    }


def evaluate_model_responses(
    model_dir: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Đánh giá tất cả task responses của một model.
    Thêm điểm theo category.
    Rule Application: tách thành conclusion (accuracy) và reasoning (ROUGE-L).
    """
    model_name = model_dir.name
    print(f"\n{'='*60}")
    print(f"📊 ĐÁNH GIÁ MODEL: {model_name}")
    print(f"{'='*60}")
    
    task_files = list(model_dir.glob("*.json"))
    all_results = {}
    
    # 5 Categories:
    # - issue-spotting: accuracy
    # - rule-conclusion: accuracy của rule tasks (answer)
    # - rule-application: ROUGE-L của rule tasks (reasoning)
    # - interpretation: clause_type (accuracy) + entity_extraction (ROUGE-L) → gộp
    # - rhetorical-understanding: accuracy
    
    category_data = {
        "issue-spotting": {"correct": 0, "total": 0},
        "rule-conclusion": {"correct": 0, "total": 0},
        "rule-application": {"total_score": 0.0, "count": 0},
        "interpretation": {"correct": 0, "total": 0, "rouge_score": 0.0, "rouge_count": 0},
        "rhetorical-understanding": {"correct": 0, "total": 0},
        "rule-recall": {"correct": 0, "total": 0},
    }
    
    for task_file in task_files:
        task_name = task_file.stem
        print(f"\n📋 Task: {task_name}")
        
        result = evaluate_task_file(task_file)
        all_results[task_name] = result
        
        # Phân loại task
        task_category = TASK_CATEGORIES.get(task_name)
        
        if task_category == "issue-spotting":
            category_data["issue-spotting"]["correct"] += result["correct"]
            category_data["issue-spotting"]["total"] += result["total_samples"]
        
        elif task_category == "rule":
            # Rule Conclusion: accuracy
            category_data["rule-conclusion"]["correct"] += result["correct"]
            category_data["rule-conclusion"]["total"] += result["total_samples"]
            # Rule Application: reasoning ROUGE-L
            if result.get("avg_reasoning_rouge_l") is not None:
                category_data["rule-application"]["total_score"] += result["avg_reasoning_rouge_l"] * result["total_samples"]
                category_data["rule-application"]["count"] += result["total_samples"]
        
        elif task_category == "interpretation":
            # clause_type_identification: dùng accuracy
            if task_name == "clause_type_identification":
                category_data["interpretation"]["correct"] += result["correct"]
                category_data["interpretation"]["total"] += result["total_samples"]
            # legal_entity_extraction: dùng ROUGE-L
            elif task_name == "legal_entity_extraction":
                if result.get("avg_entity_rouge_l") is not None:
                    category_data["interpretation"]["rouge_score"] += result["avg_entity_rouge_l"] * result["total_samples"]
                    category_data["interpretation"]["rouge_count"] += result["total_samples"]
        
        elif task_category == "rhetorical-understanding":
            category_data["rhetorical-understanding"]["correct"] += result["correct"]
            category_data["rhetorical-understanding"]["total"] += result["total_samples"]
        
        elif task_category == "rule-recall":
            category_data["rule-recall"]["correct"] += result["correct"]
            category_data["rule-recall"]["total"] += result["total_samples"]
        
        print(f"   Accuracy: {result['correct']}/{result['total_samples']} ({result['accuracy']*100:.1f}%)")
        if result.get('avg_reasoning_rouge_l') is not None:
            print(f"   Reasoning ROUGE-L: {result['avg_reasoning_rouge_l']*100:.1f}%")
        if result.get('avg_entity_rouge_l') is not None:
            print(f"   Entity ROUGE-L: {result['avg_entity_rouge_l']*100:.1f}%")
        if result['json_extraction_failed'] > 0:
            print(f"   ⚠ JSON failed: {result['json_extraction_failed']}")
    
    # Tính tổng
    total_correct = sum(r["correct"] for r in all_results.values())
    total_samples = sum(r["total_samples"] for r in all_results.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Tính điểm trung bình cho từng category (MỖI CATEGORY CHỈ CÓ 1 SỐ)
    category_scores = {}
    print("\n📁 THEO CATEGORY:")
    
    for cat in CATEGORY_NAMES:
        data = category_data.get(cat, {})
        
        # Issue-spotting, Rule Conclusion, Rhetorical Understanding, Rule Recall: accuracy
        if cat in ["issue-spotting", "rule-conclusion", "rhetorical-understanding", "rule-recall"]:
            if data.get("total", 0) > 0:
                score = data["correct"] / data["total"]
                category_scores[cat] = score
                print(f"   {CATEGORY_NAMES[cat]}: {score*100:.1f}%")
        
        # Rule Application: ROUGE-L của reasoning
        elif cat == "rule-application":
            if data.get("count", 0) > 0:
                score = data["total_score"] / data["count"]
                category_scores[cat] = score
                print(f"   {CATEGORY_NAMES[cat]}: {score*100:.1f}% (ROUGE-L)")
        
        # Interpretation: trung bình clause_type (accuracy) + entity (ROUGE-L)
        elif cat == "interpretation":
            scores = []
            if data.get("total", 0) > 0:
                acc = data["correct"] / data["total"]
                scores.append(acc)
            if data.get("rouge_count", 0) > 0:
                rouge = data["rouge_score"] / data["rouge_count"]
                scores.append(rouge)
            if scores:
                score = sum(scores) / len(scores)
                category_scores[cat] = score
                print(f"   {CATEGORY_NAMES[cat]}: {score*100:.1f}%")
    
    # Tính overall = trung bình các category scores
    overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0
    
    # Lưu kết quả
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_evaluation.json"
    
    summary = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "overall": overall_score,  # Trung bình các category scores
        "category_scores": category_scores,
        "tasks": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ OVERALL: {overall_score*100:.1f}% (trung bình 5 categories)")
    print(f"💾 Saved: {output_file}")
    print(f"{'='*60}")
    
    return summary


def create_comparison_table(all_summaries: List[Dict[str, Any]], output_dir: Path):
    """
    Tạo bảng so sánh tổng hợp giữa các models.
    """
    if not all_summaries:
        return
    
    # Thu thập tất cả task names
    all_tasks = set()
    for s in all_summaries:
        all_tasks.update(s.get("tasks", {}).keys())
    
    # Tạo comparison data
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models": [s["model_name"] for s in all_summaries],
        "overall": {},
        "by_category": {},
        "by_task": {}
    }
    
    # Overall scores - trung bình các category scores
    for s in all_summaries:
        comparison["overall"][s["model_name"]] = s["overall"]
    
    # Category scores - mỗi category 1 số duy nhất
    for cat in CATEGORY_NAMES:
        comparison["by_category"][CATEGORY_NAMES[cat]] = {}
        for s in all_summaries:
            cat_score = s.get("category_scores", {}).get(cat)
            if cat_score is not None:
                comparison["by_category"][CATEGORY_NAMES[cat]][s["model_name"]] = cat_score
    
    # Task scores - chi tiết cho từng task
    rule_application_tasks = ["judgment_outcome_prediction", "numerical_constraint_check", "eligibility_logic_verification"]
    
    for task in sorted(all_tasks):
        comparison["by_task"][task] = {}
        for s in all_summaries:
            task_data = s.get("tasks", {}).get(task)
            if task_data:
                # Rule Application tasks: ghi cả conclusion và reasoning
                if task in rule_application_tasks:
                    comparison["by_task"][task][s["model_name"]] = {
                        "conclusion": task_data["accuracy"],
                        "reasoning": task_data.get("avg_reasoning_rouge_l")
                    }
                # Entity extraction: ghi ROUGE-L
                elif task == "legal_entity_extraction":
                    comparison["by_task"][task][s["model_name"]] = {
                        "entity_rouge_l": task_data.get("avg_entity_rouge_l")
                    }
                # Các task khác: chỉ ghi accuracy
                else:
                    comparison["by_task"][task][s["model_name"]] = task_data["accuracy"]
    
    # Lưu file JSON
    output_file = output_dir / "comparison_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # Tạo file Excel (CSV)
    excel_file = output_dir / "comparison_summary.csv"
    
    # Header: Model, Category1, Category2, ..., Overall
    categories = list(CATEGORY_NAMES.values())
    header = ["Model"] + categories + ["Overall"]
    
    # Rows: mỗi model 1 hàng
    rows = []
    for s in all_summaries:
        model_name = s["model_name"]
        row = [model_name]
        
        # Category scores (%)
        for cat in CATEGORY_NAMES:
            cat_score = s.get("category_scores", {}).get(cat)
            if cat_score is not None:
                row.append(f"{cat_score*100:.2f}%")
            else:
                row.append("N/A")
        
        # Overall (%)
        overall = s.get("overall", 0)
        row.append(f"{overall*100:.2f}%")
        rows.append(row)
    
    # Ghi file CSV
    import csv
    with open(excel_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"📊 Excel saved: {excel_file}")
    
    # In bảng so sánh
    print("\n" + "="*80)
    print("📊 BẢNG SO SÁNH CHI TIẾT")
    print("="*80)
    
    # Header
    models = [s["model_name"] for s in all_summaries]
    header = f"{'Task/Category':<40}"
    for m in models:
        short_name = m.split("_")[-1][:15]
        header += f"{short_name:>15}"
    print(header)
    print("-"*80)
    
    # Overall
    row = f"{'OVERALL':<40}"
    for s in all_summaries:
        row += f"{s['overall']*100:>14.1f}%"
    print(row)
    print("-"*40)
    
    # By Category - mỗi category 1 số duy nhất
    print("\n📁 THEO CATEGORY:")
    for cat in CATEGORY_NAMES:
        cat_name = CATEGORY_NAMES[cat]
        row = f"  {cat_name:<38}"
        
        for s in all_summaries:
            cat_score = s.get("category_scores", {}).get(cat)
            if cat_score is not None:
                # Thêm (ROUGE-L) cho các category dùng ROUGE-L
                if cat in ["rule-application", "entity-extraction"]:
                    row += f"{cat_score*100:>14.1f}%"
                else:
                    row += f"{cat_score*100:>14.1f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    # By Task
    print("\n📋 THEO TASK:")
    for task in sorted(all_tasks):
        row = f"  {task:<38}"
        for s in all_summaries:
            task_data = s.get("tasks", {}).get(task)
            if task_data:
                row += f"{task_data['accuracy']*100:>14.1f}%"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("\n" + "="*80)
    print(f"💾 Saved: {output_file}")
    
    return comparison


def evaluate_all_models(
    responses_dir: Path,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Đánh giá tất cả models trong responses directory.
    Tạo bảng so sánh chi tiết.
    """
    print("\n" + "="*60)
    print("🚀 RESPONSE EVALUATOR")
    print("="*60)
    print(f"Input: {responses_dir}")
    print(f"Output: {output_dir}")
    
    model_dirs = [d for d in responses_dir.iterdir() if d.is_dir()]
    all_summaries = []
    
    for model_dir in model_dirs:
        summary = evaluate_model_responses(model_dir, output_dir)
        all_summaries.append(summary)
    
    # Tạo bảng so sánh
    if all_summaries:
        create_comparison_table(all_summaries, output_dir)
    
    return all_summaries


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM responses from saved files")
    parser.add_argument("--responses-dir", type=str, 
                       default="../responses/kaggle/working/responses",
                       help="Directory containing model response folders")
    parser.add_argument("--output-dir", type=str,
                       default="../evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    
    if not responses_dir.exists():
        print(f"❌ Responses directory not found: {responses_dir}")
        exit(1)
    
    evaluate_all_models(responses_dir, output_dir)
