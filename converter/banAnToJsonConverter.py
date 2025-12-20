"""
BanAn to JSON Converter (Mistral Vision Version - Parallel)
Đọc toàn bộ file PDF trong folder BanAn bằng Mistral Vision API và chuyển thành file JSON.
Xử lý song song để tăng tốc.

Yêu cầu:
- pip install mistralai pymupdf
"""

import os
import re
import json
import base64
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from mistralai import Mistral
    import fitz  # PyMuPDF
except ImportError:
    print("Cần cài đặt: pip install mistralai pymupdf")
    exit(1)

# ============ CẤU HÌNH ============
API_KEY = "C4vlFAWlSKGFsYfT5juI4kP1gb2U4XSP"
MAX_WORKERS = 5  # Số file xử lý song song
# ==================================

# Khởi tạo Mistral
client = Mistral(api_key=API_KEY)
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def extract_id_from_text(text: str) -> str:
    """Tìm ID từ text."""
    if not text:
        return ""
    
    id_pattern = r'(\d+\s*/\s*\d{4}\s*/\s*[A-ZĐa-zđ]+\s*-?\s*[A-ZĐa-zđ]*)'
    
    keyword_patterns = [
        r'[Ss][oố][:\s]+' + id_pattern,
        r'[Bb][aả]n [aá]n s[oố][:\s]*' + id_pattern,
        r'[Qq]uy[eế]t [dđ][iị]nh s[oố][:\s]*' + id_pattern,
    ]
    
    for pattern in keyword_patterns:
        match = re.search(pattern, text)
        if match:
            raw_id = match.group(1).strip()
            clean_id = re.sub(r'\s+', '', raw_id)
            return clean_id
    
    match = re.search(id_pattern, text)
    if match:
        raw_id = match.group(1).strip()
        clean_id = re.sub(r'\s+', '', raw_id)
        return clean_id
    
    return ""


def ocr_single_page(img_base64: str, page_num: int, max_retries: int = 3) -> str:
    """OCR một trang."""
    prompt = "Trích xuất toàn bộ văn bản từ hình ảnh này. Chỉ trả về văn bản thuần túy, không thêm giải thích."
    
    for retry in range(max_retries):
        try:
            response = client.chat.complete(
                model="pixtral-large-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{img_base64}"}
                        ]
                    }
                ]
            )
            return response.choices[0].message.content if response.choices else ""
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait_time = 5 * (retry + 1)
                time.sleep(wait_time)
            else:
                raise e
    return ""


def ocr_pdf_with_mistral(pdf_path: str, file_idx: int, total: int) -> dict:
    """OCR file PDF sử dụng Mistral Vision API."""
    filename = Path(pdf_path).name
    safe_print(f"[{file_idx}/{total}] Bắt đầu: {filename}")
    
    try:
        doc = fitz.open(pdf_path)
        
        # Chuẩn bị tất cả ảnh trước
        pages_data = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            pages_data.append((i, img_base64))
        
        doc.close()
        
        # OCR song song các trang
        all_text = [""] * len(pages_data)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(ocr_single_page, img_b64, page_num): page_num 
                for page_num, img_b64 in pages_data
            }
            
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    all_text[page_num] = future.result()
                except Exception as e:
                    all_text[page_num] = ""
        
        # Xử lý kết quả
        first_page_text = all_text[0] if all_text else ""
        combined_text = " ".join(all_text)
        
        if not combined_text.strip():
            safe_print(f"[{file_idx}/{total}] ❌ {filename}: Không OCR được")
            return {"file": filename, "error": "Không OCR được nội dung"}
        
        doc_id = extract_id_from_text(first_page_text)
        
        full_text = combined_text
        full_text = re.sub(r'[\n\r\t]+', ' ', full_text)
        full_text = re.sub(r'\s{2,}', ' ', full_text)
        full_text = full_text.strip()
        
        if doc_id:
            safe_print(f"[{file_idx}/{total}] ✓ {filename} -> ID: {doc_id}")
        else:
            safe_print(f"[{file_idx}/{total}] ⚠ {filename}: Không tìm thấy ID")
        
        return {"id": doc_id, "data": full_text}
        
    except Exception as e:
        safe_print(f"[{file_idx}/{total}] ❌ {filename}: {str(e)}")
        return {"file": filename, "error": str(e)}


def convert_banan_to_json(input_folder: str, output_file: str = None):
    """Đọc tất cả file PDF trong folder bằng Mistral Vision và xuất ra file JSON."""
    input_path = Path(input_folder)
    
    if output_file is None:
        output_file = input_path / "banan_data.json"
    else:
        output_file = Path(output_file)
    
    pdf_files = sorted(list(input_path.glob("*.pdf")))
    print(f"\nTìm thấy {len(pdf_files)} file PDF")
    print(f"Sử dụng Mistral Vision API (song song {MAX_WORKERS} file)")
    print("-" * 60)
    
    results = []
    errors = []
    
    # Xử lý song song
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(ocr_pdf_with_mistral, str(pdf_file), idx, len(pdf_files)): pdf_file
            for idx, pdf_file in enumerate(pdf_files, 1)
        }
        
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)
    
    # Lưu ra file JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("-" * 60)
    print(f"✓ Đã lưu {len(results)} bản ghi vào: {output_file}")
    
    if errors:
        print(f"\n⚠ Có {len(errors)} file lỗi:")
        for err in errors:
            print(f"  - {err['file']}: {err['error']}")
    
    with_id = sum(1 for r in results if r.get('id'))
    without_id = len(results) - with_id
    print(f"\nThống kê:")
    print(f"  - Có ID: {with_id}/{len(results)}")
    print(f"  - Không có ID: {without_id}/{len(results)}")
    
    return results


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    banan_folder = script_dir.parent / "BanAn"
    
    if not banan_folder.exists():
        print(f"Không tìm thấy folder: {banan_folder}")
        exit(1)
    
    convert_banan_to_json(str(banan_folder))
