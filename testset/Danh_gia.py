# evaluator.py
import json
import ollama
from typing import Dict, List
import time

# ================== CẤU HÌNH ==================
LLM_MODEL = "gemma3:4b"  # Hoặc: phi3, qwen:1.5b, llama3:8b
INPUT_FILE = "test_set_giao_duc.json"  # File testset của bạn
OUTPUT_FILE = "evaluation_report.json"

# ================== PROMPT ĐÁNH GIÁ (LLM-AS-JUDGE) ==================
EVAL_PROMPT = """
Bạn là **giám khảo pháp lý AI**. Hãy đánh giá **một cặp Q&A** từ testset.

**Câu hỏi**: {question}
**Câu trả lời**: {answer}
**Tham chiếu luật**: {reference}

**Yêu cầu đánh giá**:
1. `answer` có **chính xác, đầy đủ, bám sát reference** không?
2. `reasoning_level` có **phù hợp** không?
   - 0: tra cứu trực tiếp
   - 1: ghép 2-3 thông tin
   - 2: suy luận phức tạp (>3 bước)
3. `insufficient_context` có **đúng** không? (nếu luật không có → true)
4. Có **lỗi ngữ pháp, lặp từ, không trung tính** không?

**Đầu ra JSON** (không thêm chữ):
{{
  "is_correct": true/false,
  "reasoning_level_correct": true/false,
  "insufficient_correct": true/false,
  "issues": ["lỗi 1", "lỗi 2"],
  "score": 0-10,
  "comment": "giải thích ngắn"
}}
"""

# ================== HÀM ĐÁNH GIÁ MỘT CÂU ==================
def evaluate_item(item: Dict) -> Dict:
    prompt = EVAL_PROMPT.format(
        question=item["question"],
        answer=item["answer"],
        reference=item["reference"]
    )

    for _ in range(3):  # Retry 3 lần
        try:
            response = ollama.generate(model=LLM_MODEL, prompt=prompt)
            text = response['response'].strip()

            # Lấy JSON từ output
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("Không tìm thấy JSON")

            json_str = text[start:end]
            eval_result = json.loads(json_str)

            # Gắn ID
            eval_result['id'] = item['id']
            return eval_result

        except Exception as e:
            print(f"Lỗi đánh giá {item['id']}: {e}")
            time.sleep(1)

    # Nếu thất bại
    return {
        "id": item["id"],
        "is_correct": False,
        "issues": ["LLM không trả JSON hợp lệ"],
        "score": 0,
        "comment": "Lỗi xử lý"
    }

# ================== CHẠY ĐÁNH GIÁ TOÀN BỘ ==================
def run_evaluation():
    print(f"Đang tải testset từ: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        testset = json.load(f)

    print(f"Tìm thấy {len(testset)} câu hỏi. Bắt đầu đánh giá bằng {LLM_MODEL}...")

    results = []
    for i, item in enumerate(testset):
        print(f"[{i+1}/{len(testset)}] Đang đánh giá: {item['id']}")
        result = evaluate_item(item)
        results.append(result)
        time.sleep(0.5)  # Tránh quá tải

    # Lưu kết quả
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Báo cáo tổng quan
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct'))
    avg_score = sum(r.get('score', 0) for r in results) / total

    print("\n" + "="*50)
    print("BÁO CÁO ĐÁNH GIÁ")
    print("="*50)
    print(f"Tổng câu: {total}")
    print(f"Đúng (answer): {correct}/{total} ({correct/total:.1%})")
    print(f"Điểm trung bình: {avg_score:.2f}/10")
    print(f"Kết quả lưu tại: {OUTPUT_FILE}")
    print("="*50)

# ================== CHẠY ==================
if __name__ == "__main__":
    run_evaluation()