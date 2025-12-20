"""
Response Generator - Chỉ generate responses từ LLM, không đánh giá

Usage:
    python responseGenerator.py --model Qwen/Qwen2.5-7B-Instruct --task all
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import từ config (settings) và llmClient (functions)
from config import BENCHMARK_MODELS, TASK_NAMES, DATA_DIR, OUTPUT_DIR
from llmClient import call_llm, unload_model


# ============================================================
# PROMPT - Import từ taskEvaluators để tránh duplicate
# ============================================================

def get_prompt_creator(task_name: str):
    """Lấy hàm tạo prompt cho task từ taskEvaluators."""
    try:
        from taskEvaluators import get_evaluator
        create_prompt_fn, _ = get_evaluator(task_name)
        return create_prompt_fn
    except ImportError:
        # Fallback nếu không import được
        return None


def create_prompt_fallback(task_name: str, instruction: str, question: str) -> str:
    """Fallback prompt nếu không import được từ taskEvaluators."""
    return f"""Bạn là chuyên gia pháp lý Việt Nam.
HƯỚNG DẪN: {instruction}
CÂU HỎI: {question}
Trả lời theo JSON format."""


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_responses(
    data_dir: Path,
    model_name: str,
    output_dir: Path,
    tasks: list = None,
    max_samples: Optional[int] = None,
    delay: float = 0.5
):
    """
    Generate responses cho tất cả tasks.
    
    Args:
        data_dir: Thư mục chứa file task JSON
        model_name: Tên model (vd: "Qwen/Qwen2.5-7B-Instruct")
        output_dir: Thư mục output
        tasks: Danh sách task cần chạy
        max_samples: Giới hạn số samples mỗi task
        delay: Delay giữa các request
    """
    if tasks is None or tasks == ["all"]:
        tasks = TASK_NAMES
    
    # Tạo output directory
    model_safe_name = model_name.replace("/", "_").replace(":", "_")
    model_output_dir = output_dir / model_safe_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 GENERATE RESPONSES")
    print(f"   Model: {model_name}")
    print(f"   Output: {model_output_dir}")
    print(f"{'='*60}")
    
    for task_name in tasks:
        print(f"\n📋 Task: {task_name}")
        
        # Load task data
        task_file = data_dir / f"{task_name}.json"
        if not task_file.exists():
            print(f"  ⚠ Không tìm thấy file: {task_file}")
            continue
        
        with open(task_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
        
        instruction = task_data.get("task_config", {}).get("instruction", "")
        samples = task_data.get("data_content", [])
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"  Số samples: {len(samples)}")
        
        # Lấy hàm tạo prompt từ taskEvaluators
        create_prompt_fn = get_prompt_creator(task_name)
        
        responses = []
        for i, sample in enumerate(samples, 1):
            question = sample.get("question", "")
            
            # Tạo prompt
            if create_prompt_fn:
                prompt = create_prompt_fn(instruction, question)
            else:
                prompt = create_prompt_fallback(task_name, instruction, question)
            
            print(f"  [{i}/{len(samples)}] Generating...", end=" ", flush=True)
            
            # Generate response - chỉ cần truyền model
            response = call_llm(prompt, model_name)
            
            print("✓" if "[ERROR]" not in response else "✗")
            
            responses.append({
                "id": i,
                "question": question,
                "reference": sample,
                "model_response": response
            })
            
            time.sleep(delay)
        
        # Lưu responses
        output_file = model_output_dir / f"{task_name}.json"
        output_data = {
            "task_name": task_name,
            "model_name": model_name,
            "total_samples": len(responses),
            "timestamp": datetime.now().isoformat(),
            "responses": responses
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 Saved: {output_file.name}")
    
    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT! Responses lưu tại: {model_output_dir}")
    print(f"{'='*60}")


# ============================================================
# RUN ALL BENCHMARK MODELS - Chạy tất cả 5 models từ config
# ============================================================

def run_all_benchmark_models(
    data_dir: Path = Path("./data"),
    output_dir: Path = Path("./responses"),
    max_samples: Optional[int] = None,
    delay: float = 0.5
):
    """
    Chạy tất cả BENCHMARK_MODELS từ config.py.
    Mỗi model: Load → Generate responses → Unload.
    """
    print("="*60)
    print("🚀 RUN ALL BENCHMARK MODELS")
    print(f"   Models: {len(BENCHMARK_MODELS)}")
    print(f"   Tasks: {len(TASK_NAMES)}")
    print("="*60)
    
    for idx, model_name in enumerate(BENCHMARK_MODELS, 1):
        print(f"\n{'='*60}")
        print(f"📦 MODEL {idx}/{len(BENCHMARK_MODELS)}: {model_name}")
        print("="*60)
        
        # Generate responses
        generate_responses(
            data_dir=data_dir,
            model_name=model_name,
            output_dir=output_dir,
            max_samples=max_samples,
            delay=delay
        )
        
        # Unload model
        unload_model()
    
    print("\n" + "="*60)
    print("✅ HOÀN TẤT TẤT CẢ BENCHMARK MODELS!")
    print(f"   Output: {output_dir}")
    print("="*60)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM responses for legal benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Tên model (vd: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name hoặc 'all'")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Thư mục chứa task data")
    parser.add_argument("--output-dir", type=str, default="./responses",
                        help="Thư mục output")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Số samples tối đa mỗi task")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay giữa các request (giây)")
    
    args = parser.parse_args()
    
    tasks = [args.task] if args.task != "all" else None
    
    generate_responses(
        data_dir=Path(args.data_dir),
        model_name=args.model,
        output_dir=Path(args.output_dir),
        tasks=tasks,
        max_samples=args.max_samples,
        delay=args.delay
    )

        max_samples=args.max_samples,
        delay=args.delay
    )

