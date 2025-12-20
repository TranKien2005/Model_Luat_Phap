"""
Legal Benchmark Data Generator
Sinh dữ liệu benchmark từ bản án sử dụng Gemini API.

Yêu cầu:
- pip install google-generativeai
- API key từ https://aistudio.google.com/apikey
"""

import os
import re
import json
import time
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Cần cài đặt: pip install google-generativeai")
    exit(1)

# ============ CẤU HÌNH ============
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyA3SvrBgCF5boaCoG8vERAVi4CDVUd3Zzs")
MODEL_NAME = "gemini-2.0-flash"  # hoặc gemini-2.0-flash
BATCH_SIZE = 5  # Số bản án gửi mỗi lần
# ==================================

# Khởi tạo Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ============ FULL PROMPT TỪ prompt.txt ============
MASTER_PROMPT = """MASTER PROMPT: LEGAL BENCHMARK DATA GENERATOR
Bạn là một chuyên gia kỹ thuật dữ liệu pháp lý. Nhiệm vụ của bạn là dựa trên bản án được cung cấp và các Task Config dưới đây và dữ liệu về bản án được gửi để sinh ra dữ liệu benmark

QUY ĐỊNH VỀ CẤU TRÚC DỮ LIỆU
Định dạng: Trả về JSON cho từng Task.

Schema loại 1 (Dành cho Category 1, 4, 5): {"task_config": {...}, "data_content": [{"question": "...", "answer": "..."}]}

Schema loại 2 (Dành cho Category 3): {"task_config": {...}, "data_content": [{"question": "...", "answer": "...", "reasoning": "..."}]}



YÊU CẦU VỀ SỐ LƯỢNG: Nếu có thể, mỗi bản án sinh ra 2-3 samples cho mỗi task.
YÊU CẦU VỀ THỨ TỰ: Các samples trong data_content phải được sắp xếp theo đúng thứ tự các bản án được cung cấp. Ví dụ: samples từ bản án 1 phải đứng trước samples từ bản án 2, samples từ bản án 2 phải đứng trước samples từ bản án 3, v.v.
Lưu ý nếu không sinh được task nhất định từ nội dung bản án cứ để data_content rỗng []
Lưu ý tất cả các trường không có trường con viết trên một dòng ví dụ:
{
  "task_config": {"task_name": "...", "category": "...", "instruction": "..."},
  "data_content": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "...", "reasoning": "..."}
  ]
}

DANH SÁCH CÁC TASK CONFIG CHI TIẾT

═══════════════════════════════════════════════════════════════════
[TASK 1.1: General_Issue_Binary]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "general_issue_binary",
  "category": "issue-spotting",
  "instruction": "Phân tích tình huống thực tế và xác định xem có sự tồn tại của vấn đề pháp lý (Legal Issue) được nêu trong câu hỏi hay không. Trả lời duy nhất 'Có' hoặc 'Không'. Tuyệt đối không giải thích thêm."
}

CHÚ Ý: Với mỗi dạng này sinh ra một câu đúng và một câu sai. Câu sai viết trước câu đúng. Đáp án câu sai không được quá dễ đoán.
CHÚ Ý: Với dạng này issue cực kỳ ngắn gọn và bao quát một vấn đề chung không đi vào cụ thể.

VÍ DỤ:
{"question": "Tình huống: Bà L sử dụng đất nông nghiệp từ năm 1974, đất bị ngập mặn từ năm 1999 do Nhà nước làm đường gây vỡ cống, đến năm 2014 đất bị thu hồi để xây trường mẫu giáo nhưng không có quyết định thu hồi và không được bồi thường hỗ trợ. Vấn đề: Đây có phải là vấn đề tranh chấp quyền sử dụng đất không?", "answer": "Không"}
{"question": "Tình huống: Bà L sử dụng đất nông nghiệp từ năm 1974, đất bị ngập mặn từ năm 1999 do Nhà nước làm đường gây vỡ cống, đến năm 2014 đất bị thu hồi để xây trường mẫu giáo nhưng không có quyết định thu hồi và không được bồi thường hỗ trợ. Vấn đề: Đây có phải là vấn đề bồi thường khi Nhà nước thu hồi đất không?", "answer": "Có"}

═══════════════════════════════════════════════════════════════════
[TASK 1.2: Case_Type_Classification]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "case_type_classification",
  "category": "issue-spotting",
  "instruction": "Dựa trên một đoạn tóm tắt thông tin của bản án hãy phân loại bản án vào một trong các nhóm sau: [Dân sự, Hành chính, Hình sự, Kinh doanh thương mại, Lao động, hôn nhân và gia đình]. Chỉ trả lời tên nhóm. Lưu ý dân sự đây không phải là dân sự theo nghĩa rộng mà theo nghĩa hẹp hơn Dân sự (nghĩa hẹp: hợp đồng, thừa kế, bồi thường...), theo nghĩa rộng thì dân sự sẽ bao gồm dân sự theo nghĩa hẹp, kinh doanh thương mại, lao động, hôn nhân và gia đình"
}
Lưu ý: tóm tát ở đây nên chỉ tóm tắt về tình huống nên tránh những thông tin trong bản án có thể khiến mô hình nhận trực tiếp ra loại vị dụ kiện hành chính.
LƯU Ý: Yêu cầu mô hình tự tóm tắt không cần quá dài chỉ cần đủ ý chính khoảng 5 đến 6 câu.

Ví dụ:
{
            "question": "Tóm tắt bản án: Ông Nguyễn Ngọc T và bà Nguyễn Thị Mỹ H yêu cầu Tòa án công nhận thuận tình ly hôn và thỏa thuận nuôi con. Hai người kết hôn năm 2001, có hai con chung đã trưởng thành. Hai bên xác định tình cảm vợ chồng không còn, mâu thuẫn trầm trọng không thể hàn gắn. Tòa án công nhận thuận tình ly hôn theo yêu cầu.",
            "answer": "Hôn nhân và gia đình"
        }

═══════════════════════════════════════════════════════════════════
[TASK 3.1: Judgment_Outcome_Prediction]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "judgment_outcome_prediction",
  "category": "rule-application",
  "instruction": "câu hỏi cung đầy đủ rule và chi tiết về tình huống, yêu cầu đưa ra phán quyết của một vất đề luật pháp được yêu cầu câu trả lời là 1 trong 4 đáp án được question đưa ra, reasoning yêu cầu giải thích theo format Theo luật... (toàn bộ hoặc một ý của luật được cung cấp trong bản án yêu cầu nêu rõ rule chứ không phải chỉ nêu tên) mà ..(fact lấy từ tình huống có thể kèm một chút suy luận nhỏ để rõ nghĩa (.. tức là, ..)) nên/ do đó (một kết luận nhỏ hoặc kết luận trực tiếp) mà theo luật (nối tiếp thành nhiều chuỗi như vậy trong trường hợp tình huống phức tạp đến khi ra được kết luận)"
}

CHÚ Ý CHO TẤT CẢ TASK 3: Question LUÔN phải cung cấp luật và tình huống. Luật cung cấp phải đầy đủ rõ ràng không mơ hồ cắt ngắn, tình huống phải đầy đủ dữ kiện toàn bộ dữ kiện, đáp án hay reasoning không được lấy ra từ ngoài đoạn tóm tắt tình huống.

VÍ DỤ:
{"question": "Luật: Theo quy định tại Điều 30, Điều 31 Luật Khiếu nại năm 2011, cơ quan nhận đơn khiếu nại phải tiến hành đối thoại và ban hành quyết định giải quyết khiếu nại, không được ban hành công văn trả lời đơn. Tình huống: Bà Nguyễn Thị L có đơn yêu cầu ngày 15/12/2015 gửi UBND huyện N về việc lập thủ tục thu hồi đất, bồi thường thiệt hại đối với hai thửa đất bị thu hồi để xây trường mẫu giáo. UBND huyện N không tiến hành đối thoại và không ban hành quyết định giải quyết khiếu nại mà chỉ ban hành Công văn số 515/UBND-TNMT ngày 18/5/2016 trả lời đơn của bà L. Hỏi: Công văn số 515/UBND-TNMT ngày 18/5/2016 của UBND huyện N có hợp pháp không?\\nA. Hợp pháp\\nB. Không hợp pháp và phải bị hủy\\nC. Hợp pháp nhưng cần bổ sung đối thoại\\nD. Không hợp pháp nhưng không cần hủy", "answer": "B", "reasoning": "Theo Điều 30, Điều 31 Luật Khiếu nại năm 2011 cơ quan nhận đơn khiếu nại phải tiến hành đối thoại và ban hành quyết định giải quyết khiếu nại mà UBND huyện N không tiến hành đối thoại và không ban hành quyết định giải quyết khiếu nại mà chỉ ban hành công văn trả lời đơn nên công văn số 515/UBND-TNMT ngày 18/5/2016 là trái quy định pháp luật do đó công văn này phải bị hủy."}

═══════════════════════════════════════════════════════════════════
[TASK 3.2: Numerical_Constraint_Check]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "numerical_constraint_check",
  "category": "rule-application",
  "instruction": "Câu hỏi cung cấp đủ rule và chi tiết về tình huống. Thực hiện kiểm tra logic toán học và điều kiện định lượng (tuổi, thời hạn, diện tích). Câu trả lời là 1 trong 4 đáp án được question đưa ra, reasoning yêu cầu giải thích theo format Theo luật ... (toàn bộ hoặc một ý của luật được cung cấp trong bản án yêu cầu nêu rõ rule chứ không phải chỉ nêu tên) mà ..(fact lấy từ tình huống có thể kèm một chút suy luận nhỏ để rõ nghĩa (.. tức là, ..)) nên/ do đó (một kết luận nhỏ hoặc kết luận trực tiếp) mà theo luật (nối tiếp thành nhiều chuỗi như vậy trong trường hợp tình huống phức tạp đến khi ra được kết luận)"
}

Ví dụ reasoning: "Theo luật người dưới 14 tuổi không bị truy cứu trách nhiệm hình sự mà An sinh vào 2015 tức là Nam mới chỉ 10 tuổi do đó Nam không bị truy cứu trách nhiệm hình sự"

═══════════════════════════════════════════════════════════════════
[TASK 3.3: Eligibility_Logic_Verification]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "eligibility_logic_verification",
  "category": "rule-application",
  "instruction": "Kiểm tra điều kiện để được hưởng quyền lợi hoặc gánh vác nghĩa vụ. Trường 'reasoning' phải phân tích logic: (nếu quy định/ không viết vào) Theo quy định L Nếu Điều kiện A + Điều kiện B thì có Quyền lợi X mà (nêu fact từ tình huống có thể biến đổi để dễ hiểu hơn ví dụ: cũng tức là /không ghi vào) Chỉ rõ thực tế có đáp ứng đủ các điều kiện thành phần hay không, (kết luận /ko viết vào) do đó... . Câu trả lời là có hoặc không"
}

VÍ DỤ:
{"question": "Theo quy định tại khoản 1 Điều 21 Nghị định số 43/2014/NĐ-CP ngày 15/5/2014, hộ gia đình, cá nhân đang sử dụng đất ổn định, liên tục thì đủ điều kiện để được cấp Giấy chứng nhận quyền sử dụng đất. Bà Nguyễn Thị L nhận chuyển nhượng đất từ năm 1974, sử dụng liên tục đến năm 1999, từ năm 1999 đến 2014 không sử dụng được do đất bị ngập mặn vì Nhà nước làm đường gây vỡ cống dẫn đến ngập úng nhiễm mặn (lý do khách quan), bà L không từ bỏ quyền sử dụng đất và vẫn kê khai khi Nhà nước yêu cầu. Hỏi: Bà L có đủ điều kiện sử dụng đất ổn định, liên tục để được xem xét bồi thường khi Nhà nước thu hồi đất không?", "answer": "Có", "reasoning": "Theo quy định tại khoản 1 Điều 21 Nghị định số 43/2014/NĐ-CP nếu sử dụng đất ổn định, liên tục thì đủ điều kiện được cấp Giấy chứng nhận quyền sử dụng đất mà bà L sử dụng đất từ năm 1974 đến năm 1999 liên tục, từ năm 1999 đến 2014 không sử dụng được do nguyên nhân khách quan là ngập mặn do Nhà nước làm đường gây vỡ cống, bà L không từ bỏ quyền sử dụng đất và vẫn kê khai khi Nhà nước yêu cầu. Chỉ rõ thực tế bà L đáp ứng điều kiện sử dụng ổn định, liên tục do đó bà L đủ điều kiện để được xem xét bồi thường khi Nhà nước thu hồi đất."}

═══════════════════════════════════════════════════════════════════
[TASK 4.1: Clause_Type_Identification]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "clause_type_identification",
  "category": "interpretation",
  "instruction": "Xác định loại nội dung/điều khoản của đoạn văn bản pháp lý trích từ bản án. Chọn một nhãn duy nhất: [Thông tin đương sự, Nội dung vụ án, Nhận định của Tòa, Quyết định, Bất khả kháng, Bảo mật, Hiệu lực]."
}

CHÚ Ý: Question phải viết đầy đủ đoạn văn.

VÍ DỤ:
{"question": "Đoạn văn: 'Bản án này là phúc thẩm có hiệu lực pháp luật kể từ ngày tuyên án (03/3/2017).'", "answer": "Hiệu lực"}
{"question": "Đoạn văn: 'Hủy Công văn số 515/UBND-TNMT ngày 18/5/2016 của UBND huyện N, tỉnh Quảng Nam về việc trả lời đơn của bà Nguyễn Thị L và buộc UBND huyện N, tỉnh Quảng Nam lập thủ tục thu hồi đất, bồi thường thiệt hại cho bà Nguyễn Thị L đối với các thửa đất số 62 tờ bản đồ số 12 diện tích 601,4m² và thửa đất số 74 tờ bản đồ số 12 diện tích 765m² tại thôn T, xã T, huyện N, tỉnh Quảng Nam theo đúng quy định của pháp luật.'", "answer": "Quyết định"}
{"question": "Đoạn văn: 'Sau năm 1999 đến năm 2014, Nhà nước làm đường DH6 gây vỡ cống ngập úng nhiễm nước mặn, do vậy bà L không thể sử dụng diện tích đất này được nữa.'", "answer": "Nhận định của Tòa"}

═══════════════════════════════════════════════════════════════════
[TASK 4.2: Legal_Entity_Extraction]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "legal_entity_extraction",
  "category": "interpretation",
  "instruction": "Cung cấp một đoạn hay toàn bộ bản án. Câu hỏi yêu cầu trích xuất chính xác danh tính của một thực thể pháp lý trong đoạn ví dụ bị cáo, nguyên đơn, địa điểm, vật chứng, hung khí,.."
}

LƯU Ý: Đoạn bản án cần đủ dài và có nhiều thông tin khác nữa tối thiểu 10 câu.

VÍ DỤ:
{"question": "Đoạn văn: 'Vào ngày 03 tháng 3 năm 2017 tại trụ sở Tòa án nhân dân cấp cao tại Đà Nẵng xét xử phúc thẩm công khai vụ án hành chính thụ lý số: 34/2016/TLPT-HC ngày 31 tháng 10 năm 2016 về việc "Kiện quyết định hành chính, hành vi hành chính trong lĩnh vực quản lý đất đai". Do Bản án hành chính sơ thẩm số 30/2016/HC-ST ngày 20 tháng 9 năm 2016 của Tòa án nhân dân tỉnh Quảng Nam bị kháng cáo. Người khởi kiện: Bà Nguyễn Thị L, sinh năm 1936, địa chỉ thôn A, xã T, huyện N, tỉnh Quảng Nam. Người bị kiện: Ủy ban nhân dân huyện N, tỉnh Quảng Nam. Nội dung: Bà L khởi kiện yêu cầu hủy công văn số 515/UBND-TNMT ngày 18/5/2016 và buộc UBND huyện N bồi thường đất bị thu hồi xây Trường mẫu giáo BM.' Yêu cầu trích xuất danh tính người khởi kiện.", "answer": "Bà Nguyễn Thị L"}

═══════════════════════════════════════════════════════════════════
[TASK 5.1: Functional_Sentence_Labelling]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "functional_sentence_labelling",
  "category": "rhetorical-understanding",
  "instruction": "Phân loại câu văn được trích dẫn vào một trong ba nhóm chức năng lý luận: [FACT] (Dữ kiện thực tế), [LAW] (Dẫn chiếu văn bản luật), [JUDGMENT] (Phán quyết hoặc nhận định của Thẩm phán)."
}

LƯU Ý: Yêu cầu trích dẫn đủ câu văn.

VÍ DỤ:
{"question": "Câu: 'Bà Nguyễn Thị L nhận chuyển nhượng đất theo giấy tay ngày 20/4/1974 và sử dụng liên tục đến năm 1999 để trồng hoa màu.'", "answer": "[FACT]"}
{"question": "Câu: 'Áp dụng Điều 100 Luật Đất đai năm 2013 và khoản 1 Điều 21 Nghị định 43/2014/NĐ-CP quy định về cấp Giấy chứng nhận quyền sử dụng đất cho hộ gia đình, cá nhân sử dụng đất ổn định.'", "answer": "[LAW]"}
{"question": "Câu: 'Việc bà L không sử dụng đất từ năm 1999 đến 2014 là do nguyên nhân khách quan là ngập mặn vì Nhà nước làm đường gây vỡ cống nên vẫn được xác định là sử dụng đất ổn định.'", "answer": "[JUDGMENT]"}

═══════════════════════════════════════════════════════════════════
[TASK 5.2: Argument_Consistency_Check]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "argument_consistency_check",
  "category": "rhetorical-understanding",
  "instruction": "Đánh giá sự tương thích giữa lập luận của một bên và chứng cứ thực tế. Trả lời 'Nhất quán' hoặc 'Mâu thuẫn'. Câu trả lời không cần Reasoning nhưng phải phản ánh đúng logic bác bỏ của Tòa án."
}

═══════════════════════════════════════════════════════════════════
[TASK 5.3: Reasoning_Method_Detection]
═══════════════════════════════════════════════════════════════════
Task Config:
{
  "task_name": "reasoning_method_detection",
  "category": "rhetorical-understanding",
  "instruction": "Xác định phương pháp tư duy pháp lý của Thẩm phán: [Textualism] (Dựa trên câu chữ khô khan của luật) hay [Purposivism] (Dựa trên mục đích của luật và hoàn cảnh khách quan để bảo vệ công lý)."
}

VÍ DỤ:
{"question": "Cách Tòa lập luận: Mặc dù quy định pháp luật yêu cầu sử dụng đất liên tục và ổn định, nhưng Tòa xem xét hoàn cảnh thực tế là đất bị ngập mặn từ năm 1999 do Nhà nước làm đường gây vỡ cống (nguyên nhân khách quan do Nhà nước gây ra), bà L không từ bỏ quyền sử dụng đất, nên vẫn công nhận sử dụng ổn định để bảo vệ quyền lợi bồi thường hợp pháp của người dân.", "answer": "[Purposivism]"}
{"question": "Cách Tòa lập luận: Dựa trên chứng cứ chuyển nhượng đúng diện tích theo GCNQSDĐ, biên bản thẩm định xác nhận không thiếu diện tích, chỉ sai vị trí do thiếu quản lý, không có chứng cứ lấn chiếm, đất không liền kề, nên bác yêu cầu mà không xem xét hoàn cảnh khách quan khác để bảo vệ quyền lợi.", "answer": "[Textualism]"}

═══════════════════════════════════════════════════════════════════
QUY ĐỊNH VỀ CẤU TRÚC OUTPUT TỔNG QUÁT
═══════════════════════════════════════════════════════════════════

OUTPUT PHẢI LÀ MỘT MẢNG JSON HỢP LỆ chứa chính xác 10 object (mỗi task một object).

CẤU TRÚC TỔNG QUÁT:
```json
[
  {
    "task_config": {
      "task_name": "general_issue_binary",
      "category": "issue-spotting", 
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."},
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "case_type_classification",
      "category": "issue-spotting",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "judgment_outcome_prediction",
      "category": "rule-application",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "...", "reasoning": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "numerical_constraint_check",
      "category": "rule-application",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "...", "reasoning": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "eligibility_logic_verification",
      "category": "rule-application",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "...", "reasoning": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "clause_type_identification",
      "category": "interpretation",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "legal_entity_extraction",
      "category": "interpretation",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "functional_sentence_labelling",
      "category": "rhetorical-understanding",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "argument_consistency_check",
      "category": "rhetorical-understanding",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  },
  {
    "task_config": {
      "task_name": "reasoning_method_detection",
      "category": "rhetorical-understanding",
      "instruction": "..."
    },
    "data_content": [
      {"question": "...", "answer": "..."}
    ]
  }
]
```

QUY TẮC BẮT BUỘC:
1. Output PHẢI là mảng JSON hợp lệ, bắt đầu bằng [ và kết thúc bằng ]
2. Mỗi object trong mảng PHẢI có 2 trường: "task_config" và "data_content"
3. "task_config" PHẢI chứa: task_name, category, instruction
4. "data_content" là mảng các object, mỗi object có:
   - Category issue-spotting, interpretation, rhetorical-understanding: {"question": "...", "answer": "..."}
   - Category rule-application: {"question": "...", "answer": "...", "reasoning": "..."}
5. Tất cả các trường KHÔNG CÓ trường con phải viết trên MỘT DÒNG
6. Nếu không sinh được data cho task nào, để data_content là mảng rỗng []
7. KHÔNG thêm text, giải thích, hay markdown nào ngoài JSON
"""


def create_prompt_for_batch(cases: list) -> str:
    """Tạo prompt cho một batch bản án."""
    
    cases_text = ""
    for i, case in enumerate(cases, 1):
        cases_text += f"\n{'='*60}\nBẢN ÁN {i} (ID: {case.get('id', 'N/A')})\n{'='*60}\n"
        cases_text += case.get('data', '')
        cases_text += "\n"
    
    prompt = f"""{MASTER_PROMPT}

{'='*60}
CÁC BẢN ÁN CẦN XỬ LÝ:
{'='*60}
{cases_text}

YÊU CẦU: Dựa trên các bản án trên, sinh dữ liệu benchmark cho TẤT CẢ 10 task.
- Mỗi bản án nên sinh ra 2-3 samples cho mỗi task nếu có thể.
- Các samples phải được sắp xếp theo đúng thứ tự bản án (samples từ bản án 1 trước, rồi bản án 2, v.v.).
Trả về một mảng JSON chứa 10 object (mỗi task một object).
Output phải là JSON hợp lệ, bắt đầu bằng [ và kết thúc bằng ]
"""
    return prompt


def extract_json_from_response(response_text: str) -> list:
    """Trích xuất JSON từ response."""
    # Tìm JSON trong markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Thử tìm array JSON
        json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response_text.strip()
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠ Lỗi parse JSON: {e}")
        return []


def generate_benchmark_data(input_file: str, output_dir: str):
    """Sinh dữ liệu benchmark từ file JSON bản án."""
    
    # Đọc dữ liệu bản án
    with open(input_file, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    print(f"\nĐọc được {len(cases)} bản án")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)
    
    # Danh sách task names
    task_names = [
        "general_issue_binary", "case_type_classification",
        "judgment_outcome_prediction", "numerical_constraint_check",
        "eligibility_logic_verification", "clause_type_identification",
        "legal_entity_extraction", "functional_sentence_labelling",
        "argument_consistency_check", "reasoning_method_detection"
    ]
    
    # Khởi tạo kết quả
    all_results = {name: [] for name in task_names}
    
    # Xử lý theo batch
    total_batches = (len(cases) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(cases), BATCH_SIZE):
        batch = cases[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        print(f"\n[Batch {batch_num}/{total_batches}] Xử lý {len(batch)} bản án...")
        
        prompt = create_prompt_for_batch(batch)
        
        # Gọi API với retry
        for retry in range(3):
            try:
                response = model.generate_content(prompt)
                result = extract_json_from_response(response.text)
                
                if result:
                    # Merge kết quả
                    for task_result in result:
                        task_name = task_result.get('task_config', {}).get('task_name')
                        if task_name and task_name in all_results:
                            data = task_result.get('data_content', [])
                            all_results[task_name].extend(data)
                            if data:
                                print(f"  ✓ {task_name}: +{len(data)} samples")
                    break
                    
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 30 * (retry + 1)
                    print(f"  ⏳ Rate limit, đợi {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Lỗi: {e}")
                    break
        
        # Delay giữa các batch
        time.sleep(3)
    
    # Lưu kết quả
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "-" * 60)
    print("Lưu kết quả:")
    
    # Load task configs từ data folder
    data_path = Path(input_file).parent.parent / "data"
    
    for task_name, data_content in all_results.items():
        # Đọc task config từ file gốc nếu có
        original_file = data_path / f"{task_name}.json"
        if original_file.exists():
            with open(original_file, 'r', encoding='utf-8') as f:
                original = json.load(f)
                task_config = original.get('task_config', {})
        else:
            task_config = {"task_name": task_name}
        
        task_data = {
            "task_config": task_config,
            "data_content": data_content
        }
        
        output_file = output_path / f"{task_name}_generated.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ {task_name}: {len(data_content)} samples -> {output_file.name}")
    
    print("\n✓ Hoàn thành!")
    return all_results


if __name__ == "__main__":
    # Kiểm tra API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print("⚠ Vui lòng set API key:")
        print("  1. Lấy key tại: https://aistudio.google.com/apikey")
        print("  2. Set biến môi trường: $env:GEMINI_API_KEY='your_key'")
        print("  3. Hoặc sửa trực tiếp trong file này dòng 24")
        exit(1)
    
    script_dir = Path(__file__).parent
    input_file = script_dir.parent / "BanAn" / "banan_data.json"
    output_dir = script_dir.parent / "data" / "generated"
    
    if not input_file.exists():
        print(f"Không tìm thấy file: {input_file}")
        exit(1)
    
    generate_benchmark_data(str(input_file), str(output_dir))
