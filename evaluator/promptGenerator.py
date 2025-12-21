"""
Prompt Generator - Tạo prompt cho các task pháp lý

Mỗi task có một hàm tạo prompt riêng.
Sử dụng bởi responseGenerator.py để gửi đến LLM.
"""

from typing import Callable, Dict

# ============================================================
# PROMPT FUNCTIONS - Mỗi task có một hàm tạo prompt
# ============================================================

def create_prompt_general_issue_binary(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy đọc kỹ câu hỏi và trả lời "Có" hoặc "Không".
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Có"}}"""


def create_prompt_case_type_classification(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy phân loại bản án thuộc một trong các nhóm sau:
- Dân sự
- Hành chính
- Hình sự
- Kinh doanh thương mại
- Lao động  
- Hôn nhân và gia đình

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Dân sự"}}"""


def create_prompt_judgment_outcome_prediction(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là thẩm phán chuyên nghiệp. Hãy:
1. Đọc kỹ tình huống và các phương án
2. Chọn phương án đúng nhất (A, B, C hoặc D)
3. Giải thích ngắn gọn lý do dựa trên căn cứ pháp luật

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không viết thêm gì ngoài JSON

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "A", "reasoning": "Theo Điều X Luật Y, do tình huống Z nên đáp án đúng là A."}}"""


def create_prompt_numerical_constraint_check(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy:
1. Đọc kỹ các điều kiện và ràng buộc số liệu
2. Tính toán chính xác theo quy định pháp luật
3. Chọn phương án đúng (A, B, C hoặc D)
4. Giải thích phép tính và căn cứ

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không viết thêm gì ngoài JSON

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "A", "reasoning": "Theo quy định, thời hạn = X + Y = Z ngày, nên đáp án đúng là A."}}"""


def create_prompt_eligibility_logic_verification(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy:
1. Đọc kỹ tình huống và các điều kiện được nêu
2. Xác định người/tổ chức có đủ điều kiện hay không
3. Trả lời "Có" nếu đủ điều kiện, "Không" nếu không đủ
4. Giải thích ngắn gọn lý do

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không viết thêm gì ngoài JSON

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Có", "reasoning": "Người này đáp ứng đủ các điều kiện: điều kiện A, điều kiện B."}}"""


def create_prompt_clause_type_identification(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy phân loại đoạn văn bản thuộc một trong các loại sau:
- Thông tin đương sự
- Nội dung vụ án
- Nhận định của Tòa
- Quyết định
- Hiệu lực

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Nhận định của Tòa"}}"""


def create_prompt_legal_entity_extraction(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy:
1. Đọc kỹ văn bản được cung cấp
2. Trích xuất chính xác thông tin được yêu cầu (tên người, tổ chức, số liệu, v.v.)
3. Chỉ trả lời thông tin được hỏi, không thêm thông tin khác

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Nguyễn Văn A"}}"""


def create_prompt_functional_sentence_labelling(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy phân loại câu thuộc một trong các loại sau:
- [FACT]: Câu mô tả sự kiện, tình tiết thực tế
- [LAW]: Câu trích dẫn hoặc đề cập đến quy định pháp luật
- [JUDGMENT]: Câu thể hiện phán quyết, kết luận của Tòa

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "[FACT]"}}"""


def create_prompt_argument_consistency_check(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy:
1. Đọc kỹ các luận điểm trong văn bản
2. Xác định các luận điểm có nhất quán với nhau hay có mâu thuẫn
3. Trả lời "Nhất quán" nếu các luận điểm logic, hợp lý với nhau
4. Trả lời "Mâu thuẫn" nếu các luận điểm xung đột hoặc không logic

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "Nhất quán"}}"""


def create_prompt_reasoning_method_detection(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp lý. Hãy xác định phương pháp lập luận của Tòa:
- [Textualism]: Tòa giải thích theo đúng văn bản của điều luật (literal interpretation)
- [Purposivism]: Tòa giải thích theo mục đích, tinh thần của điều luật (purposive interpretation)

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "[Textualism]"}}"""


def create_prompt_vn_legal_mcq(instruction: str, question: str) -> str:
    return f"""### HƯỚNG DẪN TRẢ LỜI (Chỉ trả lời theo hướng dẫn này, tuyệt đối không làm gì khác)
{instruction}

Bạn là chuyên gia pháp luật Việt Nam. Hãy:
1. Đọc kỹ câu hỏi và các phương án
2. Chọn phương án đúng nhất (A, B, C hoặc D)

Quy tắc trả lời:
- Chỉ trả lời bằng JSON theo format bên dưới
- Không giải thích, không viết thêm gì khác

### CÂU HỎI
{question}

### VÍ DỤ CÂU TRẢ LỜI JSON
{{"answer": "A"}}"""


# ============================================================
# REGISTRY - Map task_name -> create_prompt function
# ============================================================

PROMPT_CREATORS: Dict[str, Callable] = {
    "general_issue_binary": create_prompt_general_issue_binary,
    "case_type_classification": create_prompt_case_type_classification,
    "judgment_outcome_prediction": create_prompt_judgment_outcome_prediction,
    "numerical_constraint_check": create_prompt_numerical_constraint_check,
    "eligibility_logic_verification": create_prompt_eligibility_logic_verification,
    "clause_type_identification": create_prompt_clause_type_identification,
    "legal_entity_extraction": create_prompt_legal_entity_extraction,
    "functional_sentence_labelling": create_prompt_functional_sentence_labelling,
    "argument_consistency_check": create_prompt_argument_consistency_check,
    "reasoning_method_detection": create_prompt_reasoning_method_detection,
    "vn_legal_mcq": create_prompt_vn_legal_mcq,
}


def get_prompt_creator(task_name: str) -> Callable:
    """Lấy hàm tạo prompt cho task."""
    return PROMPT_CREATORS.get(task_name)


def create_prompt(task_name: str, instruction: str, question: str) -> str:
    """Tạo prompt cho task."""
    creator = get_prompt_creator(task_name)
    if creator:
        return creator(instruction, question)
    
    # Fallback prompt
    return f"""Bạn là chuyên gia pháp lý Việt Nam.
HƯỚNG DẪN: {instruction}
CÂU HỎI: {question}
Trả lời theo JSON format: {{"answer": "..."}}"""
