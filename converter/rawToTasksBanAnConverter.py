"""
Raw AI Response to Task Files Converter
Chuyển đổi dữ liệu từ rawGenFromBanAn.json thành các file task riêng biệt.

Input: data/rawGenFromBanAn.json - mảng các mảng, mỗi phần tử là response từ AI cho một bản án
Output: data/{task_name}.json - các file task riêng biệt theo cấu trúc sampleData
"""

import json
from pathlib import Path


def load_sample_task_configs(sample_dir: Path) -> dict:
    """Load task_config từ các file mẫu trong sampleData."""
    task_configs = {}
    for json_file in sample_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            task_name = data.get('task_config', {}).get('task_name')
            if task_name:
                task_configs[task_name] = data['task_config']
    return task_configs


def convert_raw_to_tasks(raw_file: Path, output_dir: Path, sample_dir: Path):
    """
    Chuyển đổi file raw AI response thành các file task riêng biệt.
    
    Args:
        raw_file: Path đến file rawGenFromBanAn.json
        output_dir: Thư mục output cho các file task
        sample_dir: Thư mục chứa các file mẫu để lấy task_config
    """
    # Đọc file raw
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Đọc được {len(raw_data)} batch responses")
    
    # Load task_config từ sampleData
    task_configs = load_sample_task_configs(sample_dir)
    print(f"Load được {len(task_configs)} task configs từ sampleData")
    
    # Danh sách task names
    task_names = [
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
    
    # Khởi tạo kết quả - mỗi task là một danh sách data_content
    all_data = {name: [] for name in task_names}
    
    # Xử lý từng batch (mỗi batch là response cho 1 bản án)
    for batch_idx, batch in enumerate(raw_data, 1):
        if not isinstance(batch, list):
            print(f"  ⚠ Batch {batch_idx} không phải là list, bỏ qua")
            continue
            
        # Xử lý từng task object trong batch
        for task_obj in batch:
            if not isinstance(task_obj, dict):
                continue
                
            task_config = task_obj.get('task_config', {})
            task_name = task_config.get('task_name')
            data_content = task_obj.get('data_content', [])
            
            if task_name and task_name in all_data:
                # Thêm data_content vào task tương ứng
                if isinstance(data_content, list):
                    all_data[task_name].extend(data_content)
    
    # Tạo thư mục output
    output_dir.mkdir(exist_ok=True)
    
    # Ghi các file task
    print("\nGhi các file task:")
    total_samples = 0
    
    for task_name in task_names:
        data_content = all_data[task_name]
        
        # Lấy task_config - ưu tiên từ sampleData, nếu không có thì tạo default
        if task_name in task_configs:
            task_config = task_configs[task_name]
        else:
            # Fallback: lấy từ raw data nếu có
            task_config = {"task_name": task_name}
            for batch in raw_data:
                if isinstance(batch, list):
                    for task_obj in batch:
                        if task_obj.get('task_config', {}).get('task_name') == task_name:
                            task_config = task_obj['task_config']
                            break
        
        output_data = {
            "task_config": task_config,
            "data_content": data_content
        }
        
        output_file = output_dir / f"{task_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        print(f"  ✓ {task_name}: {len(data_content)} samples")
        total_samples += len(data_content)
    
    print(f"\n✓ Hoàn thành! Tổng cộng {total_samples} samples từ {len(raw_data)} bản án")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    # Đường dẫn
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    raw_file = base_dir / "data" / "rawGenFromBanAn.json"
    output_dir = base_dir / "data"
    sample_dir = base_dir / "sampleData"
    
    # Kiểm tra file tồn tại
    if not raw_file.exists():
        print(f"❌ Không tìm thấy file: {raw_file}")
        exit(1)
    
    if not sample_dir.exists():
        print(f"⚠ Không tìm thấy thư mục sampleData: {sample_dir}")
        print("  Sẽ dùng task_config từ raw data")
    
    convert_raw_to_tasks(raw_file, output_dir, sample_dir)
