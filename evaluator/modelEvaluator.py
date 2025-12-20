"""
LLM Benchmark Evaluator - Phiên bản đầy đủ
Đánh giá các mô hình ngôn ngữ lớn trên bộ benchmark pháp lý Việt Nam.

Hỗ trợ:
- Groq API (miễn phí, nhanh)
- Gemini API (Google)
- OpenAI API (GPT)
- Ollama (local models)

Usage:
    python modelEvaluator.py --model groq:llama-3.1-8b-instant --task all
    python modelEvaluator.py --model gemini:gemini-1.5-flash --task general_issue_binary
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import từ config tập trung
from config import call_llm, AVAILABLE_MODELS, DEFAULT_MODELS

# Danh sách các task
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
    "reasoning_method_detection"
]


def load_task_data(data_dir: Path, task_name: str) -> dict:
    """Load dữ liệu của một task."""
    task_file = data_dir / f"{task_name}.json"
    if not task_file.exists():
        print(f"⚠ Không tìm thấy file: {task_file}")
        return None
    
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ ANSWER CHECKING ============

def check_answer(reference: str, response: str, task_name: str) -> bool:
    """Kiểm tra câu trả lời đúng hay sai."""
    # Chuẩn hóa
    ref = reference.strip().lower()
    resp = response.strip().lower()
    
    # Task binary: Có/Không
    if task_name == "general_issue_binary":
        if ref == "có":
            return "có" in resp and "không" not in resp[:10]
        else:
            return "không" in resp
    
    # Task trắc nghiệm A/B/C/D
    if task_name in ["judgment_outcome_prediction", "numerical_constraint_check"]:
        # Tìm đáp án A/B/C/D đầu tiên trong response
        ref_letter = ref.strip()
        # Check if response starts with or contains the correct letter
        return resp.startswith(ref_letter) or f" {ref_letter}." in resp or f" {ref_letter}:" in resp or f" {ref_letter} " in resp
    
    # Task eligibility: Có/Không
    if task_name == "eligibility_logic_verification":
        if ref == "có":
            return "có" in resp[:20] and "không" not in resp[:20]
        else:
            return "không" in resp[:20]
    
    # Task classification
    if task_name == "case_type_classification":
        return ref in resp
    
    if task_name == "clause_type_identification":
        return ref in resp
    
    if task_name == "functional_sentence_labelling":
        return ref in resp
    
    if task_name == "argument_consistency_check":
        if "nhất quán" in ref:
            return "nhất quán" in resp and "mâu thuẫn" not in resp
        else:
            return "mâu thuẫn" in resp
    
    if task_name == "reasoning_method_detection":
        if "textualism" in ref:
            return "textualism" in resp
        else:
            return "purposivism" in resp
    
    # Task extraction - substring match
    if task_name == "legal_entity_extraction":
        return ref in resp or resp in ref
    
    return ref == resp


# ============ EVALUATION ============

def evaluate_task(
    data_dir: Path,
    task_name: str,
    model_type: str,
    model_name: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
    delay: float = 1.0
) -> dict:
    """Đánh giá một task cụ thể."""
    from taskEvaluators import get_evaluator
    
    print(f"\n📋 Đánh giá task: {task_name}")
    print("-" * 50)
    
    # Load task data
    task_data = load_task_data(data_dir, task_name)
    if not task_data:
        return None
    
    # Lấy instruction từ task_config
    instruction = task_data["task_config"].get("instruction", "")
    samples = task_data["data_content"]
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"  Số samples: {len(samples)}")
    
    # Get task-specific evaluator
    create_prompt_fn, evaluate_fn = get_evaluator(task_name)
    if not create_prompt_fn or not evaluate_fn:
        print(f"  ❌ Không tìm thấy evaluator cho task: {task_name}")
        return None
    
    results = []
    correct = 0
    total_conclusion_score = 0.0
    total_reasoning_score = 0.0
    reasoning_count = 0
    
    for i, sample in enumerate(samples, 1):
        question = sample.get("question", "")
        
        # Tạo prompt riêng cho task (yêu cầu JSON response)
        prompt = create_prompt_fn(instruction, question)
        
        print(f"  [{i}/{len(samples)}] Đang xử lý...", end=" ", flush=True)
        
        # Gọi LLM qua unified interface từ config
        model_response = call_llm(prompt, api=model_type, model=model_name)
        
        # Đánh giá bằng hàm riêng của task - trả về Dict
        eval_result = evaluate_fn(sample, model_response)
        
        is_correct = eval_result["conclusion_correct"]
        conclusion_score = eval_result["conclusion_score"]
        reasoning_score = eval_result["reasoning_score"]
        details = eval_result["details"]
        
        if is_correct:
            correct += 1
            print("✓")
        else:
            print("✗")
        
        total_conclusion_score += conclusion_score
        
        # Tính reasoning score nếu có
        if reasoning_score is not None:
            total_reasoning_score += reasoning_score
            reasoning_count += 1
        
        results.append({
            "question": question,
            "reference": sample,
            "model_response": model_response,
            "conclusion_correct": is_correct,
            "conclusion_score": conclusion_score,
            "reasoning_score": reasoning_score,
            "details": details
        })
        
        # Delay để tránh rate limit
        time.sleep(delay)
    
    accuracy = correct / len(samples) if samples else 0
    avg_conclusion_score = total_conclusion_score / len(samples) if samples else 0
    avg_reasoning_score = total_reasoning_score / reasoning_count if reasoning_count > 0 else None
    
    # Lưu kết quả chi tiết
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_name}_{model_name.replace('/', '_').replace(':', '_')}_results.json"
    
    output_data = {
        "task_name": task_name,
        "model_type": model_type,
        "model_name": model_name,
        "total_samples": len(samples),
        "correct": correct,
        "accuracy": accuracy,
        "avg_conclusion_score": avg_conclusion_score,
        "avg_reasoning_score": avg_reasoning_score,
        "timestamp": datetime.now().isoformat(),
        "responses": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n  📊 Kết quả:")
    print(f"     Conclusion: {correct}/{len(samples)} ({accuracy*100:.1f}%)")
    if avg_reasoning_score is not None:
        print(f"     Reasoning: {avg_reasoning_score*100:.1f}%")
    print(f"  💾 Saved: {output_file.name}")
    
    return output_data


def run_full_evaluation(
    data_dir: Path,
    model_type: str,
    model_name: str,
    output_dir: Path,
    tasks: list = None,
    max_samples: Optional[int] = None,
    delay: float = 1.0
):
    """Chạy đánh giá trên tất cả các task."""
    if tasks is None or tasks == ["all"]:
        tasks = TASK_NAMES
    
    print(f"\n{'='*60}")
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ")
    print(f"{'='*60}")
    print(f"Model: {model_type}:{model_name}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {output_dir}")
    
    all_results = {}
    
    for task_name in tasks:
        result = evaluate_task(
            data_dir, task_name, model_type, model_name, 
            output_dir, max_samples, delay
        )
        if result:
            all_results[task_name] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total_samples"]
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 TÓM TẮT KẾT QUẢ")
    print(f"{'='*60}")
    
    total_correct = sum(r["correct"] for r in all_results.values())
    total_samples = sum(r["total"] for r in all_results.values())
    
    for task, result in all_results.items():
        print(f"  {task}: {result['accuracy']*100:.1f}%")
    
    if total_samples > 0:
        print(f"\n  TỔNG: {total_correct}/{total_samples} ({total_correct/total_samples*100:.1f}%)")
    
    # Lưu summary
    summary_file = output_dir / f"summary_{model_name.replace('/', '_').replace(':', '_')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model_type": model_type,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "overall_accuracy": total_correct/total_samples if total_samples > 0 else 0
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Summary saved: {summary_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Evaluator")
    parser.add_argument("--model", type=str, required=True,
                       help="Model (format: type:name). Examples: groq:llama-3.1-8b-instant, gemini:gemini-1.5-flash")
    parser.add_argument("--task", type=str, default="all",
                       help="Task to evaluate (task name or 'all')")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per task (for testing)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    # Parse model
    if ":" in args.model:
        model_type, model_name = args.model.split(":", 1)
    else:
        model_type = args.model
        model_name = args.model
    
    # Paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "evaluation_results"
    
    # Parse tasks
    tasks = [args.task] if args.task != "all" else None
    
    run_full_evaluation(
        data_dir=data_dir,
        model_type=model_type,
        model_name=model_name,
        output_dir=output_dir,
        tasks=tasks,
        max_samples=args.max_samples,
        delay=args.delay
    )
