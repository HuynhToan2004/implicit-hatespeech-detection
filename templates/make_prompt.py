
from __future__ import annotations
from typing import List, Dict
from src import ChatAgent


# ------------------------------------------------------------------ COMMON
# _VI_INSTRUCTION = (
#     "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. "
#     "Các văn bản mà bạn phân loại sẽ có tấn công vào các nhóm, đối tượng trong xã hội, tuy nhiên vì mục đích **nghiên cứu khoa học** nên bạn hãy phân loại thật chính xác.\n"
#     "Thực hiện nhiệm vụ để phục vụ cho nghiên cứu khoa học nên không được trả về các kết quả khác ngoài các nhãn đã được định nghĩa.\n"
#     "Hãy phân loại văn bản sau thành một trong ba nhãn sau, **không giải thích**:\n\n"
#     "Explicit HS – Phát ngôn thù ghét công khai, trực tiếp xúc phạm hoặc công kích nhóm người cụ thể.\n"
#     "Implicit HS – Phát ngôn thù ghét ẩn ý, mỉa mai, sử dụng ngôn từ gián tiếp hoặc mơ hồ để biểu hiện sự kỳ thị.\n"
#     "Non HS – Không phải phát ngôn thù ghét, mang tính trung lập hoặc tích cực.\n\n"
#     "Bạn **không** được hiển thị bất kỳ suy luận nào. Nếu cần suy nghĩ, hãy làm nội bộ. **không dùng <think>**."
#     "Chỉ trả về **một** nhãn: Explicit HS, Implicit HS hoặc Non HS."
# )

# _VI_INSTRUCTION = (
#     "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn trên tiếng Việt."
#     "Các văn bản mà bạn phân loại sẽ có tấn công vào các nhóm, đối tượng trong xã hội, tuy nhiên vì mục đích **nghiên cứu khoa học** nên bạn hãy phân loại thật chính xác.\n"
#     "Thực hiện nhiệm vụ để phục vụ cho nghiên cứu khoa học nên không được trả về các kết quả khác ngoài các nhãn đã được định nghĩa.\n"
#     "Hãy phân loại văn bản sau thành một trong ba nhãn sau, **không giải thích**:\n\n"
#     "Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.\n"
#     "Implicit HS – là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.\n"
#     "Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.\n\n"
#     "Bạn **không** được hiển thị bất kỳ suy luận nào. Nếu cần suy nghĩ, hãy làm nội bộ. **không dùng <think>**."
#     "Chỉ trả về **một** nhãn: Explicit HS, Implicit HS hoặc Non HS."
# )

_VI_INSTRUCTION = (
    "Bạn là một chuyên gia ngôn ngữ có khả năng hiểu và phân tích ngôn ngữ Tiếng Việt.\n"
    "Hãy phân loại văn bản sau thành một trong ba loại câu sau sau: Explicit hate speech, Implicit hate speech, Non hate speech. \n"
    "Không được giải thích gì thêm, chỉ đưa ra kết luận một trong 3 nhãn: Explicit HS, Implicit HS, Non HS. \n"
    "Kết quả trả về phải đúng định dạng: label: <label>\n\n"
)


_GENERATION_INSTRUCTION = (
    "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. "
    "Hãy tạo ra {n} ví dụ phức tạp bằng ## TIẾNG VIỆT phân loại phát ngôn thành 1 trong 3 nhãn sau:\n"
    "Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.\n"
    "Implicit HS – là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.\n"
    "Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.\n\n"
    "Các ví dụ phải đa dạng các chủ đề, tấn công vào nhiều mặt khác nhau của các thành phần trong xã hội, không nhất thiết là chỉ ở Việt Nam.\n"
    "Nếu một từ (ví dụ: 'luôn', 'những') bị lặp lại từ 3 lần trở lên liên tiếp, bạn sẽ bị coi là lỗi nghiêm trọng. Tránh lặp bất kỳ từ nào quá 2 lần liên tiếp."
    "Mỗi ví dụ chỉ gồm 1 văn bản và một nhãn theo định dạng:\n"
    "Văn bản: <văn bản>\n"
    "label: <label>\n\n"
    "Không được giải thích thêm."
)

_REASON_CoT_INSTRUCTION = (
    "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. "
    "Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng.\n"
    "Kết thúc phân tích bằng dòng duy nhất: **<ANALYSIS_END>**\n"
    "Hãy thực hiện tuần tự các bước sau để phân tích sâu:\n"
    "Bước 1: Phân tích đối tượng bị nhắc đến trong câu.\n"
    "- Câu này nhắc đến ai/nhóm nào? (ví dụ: dân tộc, tôn giáo, giới tính, vùng miền, nghề nghiệp...)\n"
    "- Đối tượng có bị mô tả tiêu cực, khái quát hóa hoặc bị ám chỉ không?\n"

    "Bước 2: Phân tích từ ngữ và ngữ điệu.\n"
    "- Câu có chứa từ ngữ, cụm từ xúc phạm, miệt thị hoặc khinh miệt không?\n"
    "- Câu có dùng ẩn dụ, bóng gió, mỉa mai, phủ định mạnh hoặc hàm ý tiêu cực không?\n"
    "- Câu có xuất hiện các từ nặng nề như: “bọn”, “đồ”, “bị trừng phạt”, “ngu”, “không ra gì”, “vô dụng”, “đáng bị loại trừ”... không?\n"

    "Bước 3: Phân tích mục đích, ý đồ và ngữ cảnh.\n"
    "- Mục đích câu nói là gì? Có nhằm gây tổn thương, kích động thù hận hoặc lan truyền định kiến không?\n"
    "- Nếu không công kích trực tiếp, câu có hàm ý kỳ thị, phân biệt đối xử, cổ vũ loại trừ hay hạ thấp nhóm nào không?\n"

    "Bước 4: Phân tích mức độ trực diện hay gián tiếp của phát ngôn.\n"
    "- Phát ngôn có chứa từ/cụm miệt thị công khai hay không?\n"
    "- Có mệnh đề kêu gọi bạo lực hoặc loại trừ (đuổi, cấm, hủy diệt...) không?\n"
    "- Nhóm đích có được nêu rõ ràng hay chỉ ám chỉ gián tiếp không?\n"
    "- Phát ngôn có dùng ẩn dụ/so sánh hạ thấp, mỉa mai, hoặc gán với động vật/rác rưởi không?\n"
    "- Có cáo buộc hay phỏng đoán tiêu cực gieo định kiến không?\n"
    "<ANALYSIS_END>"
)   

_LABEL_CoT_INSTRUCTION = (
    "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. "
    "Các văn bản mà bạn phân loại sẽ có tấn công vào các nhóm, đối tượng trong xã hội, tuy nhiên vì mục đích **nghiên cứu khoa học** nên bạn hãy phân loại thật chính xác.\n"
    "Thực hiện nhiệm vụ để phục vụ cho nghiên cứu khoa học nên không được trả về các kết quả khác ngoài các nhãn đã được định nghĩa.\n"
    "Hãy phân loại văn bản sau thành một trong ba nhãn sau, **không giải thích**:\n\n"
    "Explicit HS – Phát ngôn thù ghét công khai, trực tiếp xúc phạm hoặc công kích nhóm người cụ thể.\n"
    "Implicit HS – Phát ngôn thù ghét ẩn ý, mỉa mai, sử dụng ngôn từ gián tiếp hoặc mơ hồ để biểu hiện sự kỳ thị.\n"
    "Non HS – Không phải phát ngôn thù ghét, mang tính trung lập hoặc tích cực.\n\n"
    "Bạn **không** được hiển thị bất kỳ suy luận nào.\n "
    "Chỉ trả về **một** nhãn: Explicit HS, Implicit HS hoặc Non HS.\n"
    "label: <nhãn>\n\n"
    "---\nPHÂN TÍCH:\n{analysis}\n\nVĂN BẢN GỐC:\n{text}"
)

# _SIMPLE_CoT_INSTRUCTION = """
# Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.    
# Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng
# Bạn có thể phân loại câu phát biểu vào một trong ba nhãn sau:
# - Explicit HS – Phát ngôn thù ghét công khai, trực tiếp xúc phạm hoặc công kích nhóm người cụ thể.
# - Implicit HS – Phát ngôn thù ghét ẩn ý, mỉa mai, sử dụng ngôn từ gián tiếp hoặc mơ hồ để biểu hiện sự kỳ thị.
# - Non HS – Không phải phát ngôn thù ghét, mang tính trung lập hoặc tích cực.
# Hãy phân tích theo từng bước rồi đưa ra câu trả lời.
# - Bước 1: Xác định nhóm được nhắc đến trong câu nói.
# - Bước 2: Phân tích từ ngữ thể hiện tính chất trong câu.
# - Bước 3: Tìm kiếm dấu hiệu kỳ thị, xúc phạm được thể hiện.
# - BƯớc 4: **PHÂN LOẠI** nhãn cho câu phát biểu thuộc một trong ba nhãn đã nêu trên.
# Kết quả trả về theo định dạng: label: <label> \n\n
# """

#### Cao nhất ở Implicit HS ở hầu hết
_SIMPLE_CoT_INSTRUCTION = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.    
Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng
Bạn **hãy** phân loại câu phát biểu vào một trong ba nhãn sau:
- Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
- Implicit HS –  là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
- Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.
Hãy phân tích theo từng bước rồi đưa ra câu trả lời.
- Bước 1: Phân tích từ ngữ bề mặt: liệt kê các từ có khả năng xúc phạm. 
- Bước 2: Phân loại loại xúc phạm: xác định các từ ngữ được dùng là xúc phạm trực tiếp hay gián tiếp. 
- Bước 3: Phân tích sắc thái ngữ nghĩa của câu dựa trên các từ có khả năng xúc phạm: xem xét câu mang sắc thái tiêu cực, trung lập hay tích cực?
- Bước 4: Phân tích mục đích: Câu nói nhằm mục đích gì? Có nhằm làm tổn thương, phỉ báng, lan truyền định kiến hoặc kích động thù ghét đối với một cá nhân hoặc nhóm người cụ thể hay không?
- Bước 5: Tổng hợp các yếu tố trên sau đó **PHÂN LOẠI** nhãn cho câu phát biểu thuộc một trong ba nhãn đã nêu trên.
Bạn **nhất định** phải trả về nhãn theo định dạng: label: <label> sau đó phải **in ra hết** các bước phân tích. \n\n
"""

##Phân tích mục đích phát ngôn
_SIMPLE_CoT_INSTRUCTION_SCEN1 = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.    
Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng
Bạn **hãy** phân loại câu phát biểu vào một trong ba nhãn sau:
- Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
- Implicit HS –  là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
- Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.
Hãy phân tích theo từng bước rồi đưa ra câu trả lời. 
- **Bước 1**: Xác định mục đích của câu phát biểu (thông tin, phản biện, xúc phạm, trêu đùa, v.v.).  
- **Bước 2**: Nếu mục đích là xúc phạm, xem xét liệu nó có trực tiếp (dùng từ ngữ rõ ràng) hay gián tiếp (ẩn ý) không.  
- **Bước 3**: Dựa trên mục đích và cách thể hiện, quyết định nhãn phù hợp và giải thích ngắn gọn.
Bạn **nhất định** phải trả về nhãn theo định dạng: label: <label> sau đó phải **in ra hết** các bước phân tích. \n\n
"""

##Phân tích từ ngữ và ngữ cảnh
#### Cao nhất ở explicit HS và Non HS ở hầu hết
_SIMPLE_CoT_INSTRUCTION_SCEN2 = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.    
Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng
Bạn **hãy** phân loại câu phát biểu vào một trong ba nhãn sau:
- Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
- Implicit HS –  là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
- Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.
Hãy phân tích theo từng bước rồi đưa ra câu trả lời. 
- **Bước 1**: Liệt kê các từ ngữ có khả năng xúc phạm trong câu (nếu có).  
- **Bước 2**: Xem xét ngữ cảnh để xác định liệu các từ ngữ đó có được dùng với ý nghĩa xúc phạm hay không (giải thích ngắn gọn).  
- **Bước 3**: Nếu có ý nghĩa xúc phạm, xác định xem nó trực tiếp hay gián tiếp.  
- **Bước 4**: Dựa trên các bước trên, quyết định nhãn phù hợp.
Bạn **nhất định** phải trả về nhãn theo định dạng: label: <label> sau đó phải **in ra hết** các bước phân tích. \n\n
"""

##Phân tích mức độ nghiêm trọng
_SIMPLE_CoT_INSTRUCTION_SCEN3 = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.    
Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng
Bạn **hãy** phân loại câu phát biểu vào một trong ba nhãn sau:
- Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
- Implicit HS –  là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
- Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.
Hãy phân tích theo từng bước rồi đưa ra câu trả lời. 
- **Bước 1**: Xác định mức độ nghiêm trọng của câu (nhẹ, trung bình, nặng) và giải thích ngắn gọn.  
- **Bước 2**: Nếu mức độ nghiêm trọng là trung bình hoặc nặng, xem xét liệu nó có trực tiếp hay gián tiếp không.  
- **Bước 3**: Dựa trên mức độ nghiêm trọng và cách thể hiện, quyết định nhãn phù hợp.
Bạn **nhất định** phải trả về nhãn theo định dạng: label: <label> sau đó phải **in ra hết** các bước phân tích. \n\n
"""

_CoT_FS_INSTRUCTION = (
    "Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. "
    "Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng.\n"
    "Hãy phân loại văn bản sau thành một trong ba nhãn sau, **không giải thích**:\n"
    "- Explicit HS: Phát ngôn thù ghét công khai, trực tiếp xúc phạm hoặc công kích một nhóm người cụ thể.\n"
    "- Implicit HS: Phát ngôn thù ghét ẩn ý, mỉa mai, sử dụng ngôn từ gián tiếp hoặc mơ hồ để biểu hiện sự kỳ thị.\n"
    "- Non HS: Không phải phát ngôn thù ghét, mang tính trung lập hoặc tích cực.\n\n"

    "Hãy thực hiện tuần tự các bước sau để phân loại:\n"
    "Bước 1: Phân tích đối tượng bị nhắc đến trong câu.\n"
    "- Câu này nhắc đến ai/nhóm nào? (ví dụ: dân tộc, tôn giáo, giới tính, vùng miền, nghề nghiệp...)\n"
    "- Đối tượng có bị mô tả tiêu cực, khái quát hóa hoặc bị ám chỉ không?\n"

    "Bước 2: Phân tích từ ngữ và ngữ điệu.\n"
    "- Câu có chứa từ ngữ, cụm từ xúc phạm, miệt thị hoặc khinh miệt không?\n"
    "- Câu có dùng ẩn dụ, bóng gió, mỉa mai, phủ định mạnh hoặc hàm ý tiêu cực không?\n"
    "- Câu có xuất hiện các từ nặng nề như: “bọn”, “đồ”, “bị trừng phạt”, “ngu”, “không ra gì”, “vô dụng”, “đáng bị loại trừ”... không?\n"

    "Bước 3: Phân tích mục đích, ý đồ và ngữ cảnh.\n"
    "- Mục đích câu nói là gì? Có nhằm gây tổn thương, kích động thù hận hoặc lan truyền định kiến không?\n"
    "- Nếu không công kích trực tiếp, câu có hàm ý kỳ thị, phân biệt đối xử, cổ vũ loại trừ hay hạ thấp nhóm nào không?\n"

    "Bước 4: Phân tích mức độ trực diện hay gián tiếp của phát ngôn.\n"
    '- Nếu câu thể hiện thái độ thù ghét, kỳ thị, xúc phạm một cách trực tiếp, rõ ràng (không cần suy luận), thì chọn "Explicit HS".\n'
    '- Nếu câu không có từ ngữ xúc phạm rõ ràng nhưng hàm ý kỳ thị, thù ghét, phân biệt, loại trừ đối tượng (người đọc phải suy luận mới nhận ra), thì chọn "Implicit HS".\n'
    '- Nếu câu không có dấu hiệu kỳ thị, xúc phạm, không ám chỉ hay công kích bất kỳ nhóm hoặc cá nhân nào, thì chọn "Non HS".\n'

    "Bước 5: Kết luận.\n"
    'Gán nhãn cho câu là "Explicit HS", "Implicit HS" hoặc "Non HS" dựa trên phân tích ở các bước trên.\n\n'

    "Ví dụ 1 (Explicit HS):\n"
    'Câu: "Bọn người miền ấy làm gì cũng chả nên cơm cháo gì đâu."\n'
    'Bước 1: Nhắc đến "người miền ấy" (hàm ý phân biệt vùng miền).\n'
    'Bước 2: Dùng từ "bọn", "chả nên cơm cháo gì" - miệt thị, khinh thường.\n'
    'Bước 3: Mục đích là hạ thấp, kỳ thị người thuộc vùng miền.\n'
    'Bước 4: Dùng từ ngữ trực diện, rõ ràng, không che giấu → Explicit HS.\n'
    'Bước 5: Nhãn: Explicit HS.\n\n'

    "Ví dụ 2 (Implicit HS):\n"
    'Câu: "Làm bạn với mấy người ấy chỉ có mà khổ cả đời."\n'
    'Bước 1: Nhắc đến "mấy người ấy" (ám chỉ một nhóm nhất định, không nói rõ ai).\n'
    'Bước 2: Không có từ ngữ thô tục, nhưng hàm ý tiêu cực, cảnh báo, miệt thị nhóm này qua cụm "chỉ có mà khổ".\n'
    'Bước 3: Mục đích là khiến người khác xa lánh, có ý phân biệt nhóm người này.\n'
    'Bước 4: Ý kỳ thị, phân biệt, nhưng gián tiếp, không dùng từ ngữ rõ ràng → Implicit HS.\n'
    'Bước 5: Nhãn: Implicit HS.\n\n'

    "Ví dụ 3 (Non HS):\n"
    'Câu: "Hôm nay thời tiết đẹp, mọi người đều vui vẻ đi làm."\n'
    'Bước 1: Không nhắc đến đối tượng cụ thể nào, chỉ nói về mọi người nói chung.\n'
    'Bước 2: Không có từ ngữ xúc phạm hay hàm ý tiêu cực.\n'
    'Bước 3: Không nhằm gây tổn thương hoặc kỳ thị.\n'
    'Bước 4: Không có dấu hiệu kỳ thị, xúc phạm hay ám chỉ phân biệt.\n'
    'Bước 5: Nhãn: Non HS.\n'
    'Sau khi hoàn tất suy luận, hãy trả lời nhãn sau đó giải thích lí do tại sao bạn lại chọn nhãn đó.\n'
    # 'Sau khi hoàn tất suy luận, hãy trả lời ngắn gọn, đảm bảo rằng trong câu trả lời có xuất hiện nhãn "Explicit HS", "Implicit HS" hoặc "Non HS".\n'
)   

_FEWSHOT_EXAMPLES: List[Dict[str, str]] = [
    {"text": "Người da đen là loài hạ đẳng, chẳng có giá trị gì trong xã hội văn minh cả.", "label": "Explicit HS"},
    {"text": "Mày nên tự tử đi cho đỡ tốn chỗ trên đời này.", "label": "Explicit HS"},
    {"text": "Người Hồi giáo không được chào đón ở đây. Chúng chỉ là một lũ đạo đức giả không muốn hòa nhập.", "label": "Explicit HS"},
    {"text": "Niềm tin của họ có thể là nguồn sức mạnh, nhưng đôi khi lại khiến họ như sống trên mây, không biết thực tế là gì.", "label": "Implicit HS"},
    {"text": "Influencer gì mà sống nhờ donate với quảng cáo xàm.", "label": "Implicit HS"},
    {"text": "Tôi không quan tâm nếu bạn thấy bị xúc phạm, lẽ ra bạn nên suy nghĩ kỹ trước khi chọn sinh ra đời.", "label": "Implicit HS"},
    {"text": "Người Do Thái mới thực sự là những người cộng sản, họ nắm quyền điều khiển cả thế giới.", "label": "Implicit HS"},
    {"text": "Mỗi người đều có quyền sống theo cách mình muốn, miễn là không làm hại ai.", "label": "Non HS"},
    {"text": "Anh ấy bảo, sao em không chuyển ra ở cùng anh nhỉ? Hoặc anh có thể dọn vào ở với em. Anh nghĩ như vậy sẽ tốt cho cả hai đấy.", "label": "Non HS"},
    {"text": "Tôi không đồng tình, nhưng tôn trọng quan điểm của bạn.", "label": "Non HS"},
]

_EX_TPL_GENERIC = (
    "Văn bản: {text}\n"
    "label: {label}\n\n"       
)

# Template riêng cho Llama-3
_EX_TPL_LLAMA3 = (
    "<|start_header_id|>user\n"
    "Văn bản: {text}\n"
    "<|end_header_id|>\n"
    "<|start_header_id|>assistant\n"
    "{label}\n"
    "<|end_header_id|>\n\n"   
)
_SUFFIX_USER_GENERIC = "Văn bản: {text}\nlabel:"  




_COT2_CONTEXT = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.  
Các văn bản được đưa vào đều chứa thông tin nhạy cảm, tuy nhiên cam kết chỉ phục vụ mục đích nghiên cứu khoa học, không dùng để công kích các đối tượng.

Hãy phân tích câu phát biểu theo các bước sau và chỉ trả về kết quả phân tích, **không đưa ra nhãn cuối cùng**:
- Bước 1: Xác định xem có cá nhân hoặc nhóm đối tượng nào trong câu nói không.
- Bước 2: Phân tích từ ngữ và cách diễn đạt trong câu phát biểu.
- Bước 3: Xác định hình thức thể hiện của câu: không có hình thức xúc phạm nào, hay xúc phạm trực tiếp hoặc gián tiếp.
- Bước 4: Phân tích mục đích: Câu nói nhằm mục đích gì? Có nhằm làm tổn thương, phỉ báng, lan truyền định kiến hoặc kích động thù ghét, v.v hay không?

Định dạng trả về:
- Bước 1: <kết quả phân tích đối tượng>
- Bước 2: <kết quả phân tích từ ngữ>
- Bước 3: <kết quả phân hình thức của câu>
- Bước 4: <kết quả phân tích mục đích của câu>
"""

_COT2_LABEL = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Bạn sẽ trả về câu trả lời là **TIẾNG VIỆT**.  
Dựa trên phân tích sau đây, hãy tổng hợp và phân loại câu phát biểu vào một trong ba nhãn sau:
- Explicit HS: Phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp, sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kỳ thị có thể nhận diện rõ ràng qua từ ngữ mà không cần suy luận ý đồ hay ngữ cảnh.
- Implicit HS: Phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực. Các biểu hiện implicit đòi hỏi suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu thông điệp kỳ thị.
- Non HS: Phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử, mang tính trung lập, tích cực, phản biện hợp lý hoặc không gây tổn hại.

**Phân tích được cung cấp**:
- Bước 1: {Bước 1}
- Bước 2: {Bước 2}
- Bước 3: {Bước 3}
- Bước 4: {Bước 4}

**Chỉ được dựa trên những phân tích trên**, không được tạo thêm bất kì phân tích nào khác
Dựa trên các phân tích trên, hãy:
1. Tổng hợp các yếu tố từ các bước, đảm bảo tham chiếu rõ ràng đến từng bước (Bước 1, Bước 2, Bước 3, Bước 4).
2. Phân loại nhãn cho câu phát biểu (Explicit HS, Implicit HS, hoặc Non HS).

**Định dạng trả về**:
- label: <label>
- Tổng hợp: <tóm tắt các yếu tố từ các bước phân tích, nêu rõ cách sử dụng từng bước để đưa ra nhãn>
"""

_COT2_CONTEXT_FEWSHOT = """
văn bản 1: "Ở bến xe, cứ thấy người nói tiếng Thanh Hóa là tôi ôm chặt ví — ‘kinh nghiệm xương máu’ đấy."
Phân tích:
Bước 1: Nhắm tới “người nói tiếng Thanh Hóa” — một nhóm sắc tộc/ngôn ngữ.
Bước 2: Cấu trúc “cứ thấy… là…” khái quát hoá thành định kiến; hành vi “ôm chặt ví” hàm ý họ dễ trộm cắp; dấu nháy “kinh nghiệm xương máu” mang sắc thái mỉa mai/hợp thức hóa định kiến; không dùng từ chửi thẳng.
Bước 3: Xúc phạm gián tiếp: hàm ý quy chụp tật xấu cho cả nhóm qua gợi ý, mỉa mai và khái quát hoá.
Bước 4: Mục đích củng cố nghi ngờ mặc định đối với một cộng đồng, gieo sợ hãi và khuyến khích né tránh/kỳ thị dù không kêu gọi công khai.

văn bản 2: "Tôi phản đối việc trường đại học ưu tiên ứng viên có điểm phỏng vấn cao hơn điểm học thuật; tiêu chí nên minh bạch và dựa trên số liệu."
Bước 1: Không nhắm vào cá nhân hay nhóm được bảo vệ; đối tượng là một chính sách tuyển sinh.
Bước 2: Ngôn từ trung tính, lập luận (“phản đối”, “nên”, “minh bạch”, “dựa trên số liệu”) không có từ miệt thị, sỉ nhục hay gán nhãn.
Bước 3: Không có hình thức xúc phạm nào.
Bước 4: Mục đích là phản biện chính sách và đề xuất cải thiện; không nhằm làm tổn thương hay lan truyền định kiến.

văn bản 3: Bọn người Hồi giáo theo đạo Hồi là lũ khủng bố, cấm họ bén mảng tới khu này.”
Bước 1: Nhắm tới “người theo đạo Hồi” — một nhóm tôn giáo (nhóm được bảo vệ).
Bước 2: Từ ngữ “bọn”, “lũ” có sắc thái phi nhân hoá; “bọn người Hồi giáo” là từ xúc phạm trực diện; “khủng bố” là gán nhãn tội phạm; mệnh lệnh “cấm họ…” thúc đẩy loại trừ.
Bước 3: Xúc phạm trực tiếp: dùng từ ngữ mang tính miệt thị rõ ràng và kêu gọi loại trừ.
Bước 4: Mục đích thể hiện sự hạ nhục và gắn định kiến nguy hiểm với một tôn giáo; có khả năng kích động kỳ thị và hành vi phân biệt đối xử.
"""



_COT2_LABEL_FEWSHOT = """
văn bản 1: "Ở bến xe, cứ thấy người nói tiếng Thanh Hóa là tôi ôm chặt ví — ‘kinh nghiệm xương máu’ đấy."
Phân tích:
Bước 1: Nhắm tới “người nói tiếng Thanh Hóa” — một nhóm sắc tộc/ngôn ngữ.
Bước 2: Cấu trúc “cứ thấy… là…” khái quát hoá thành định kiến; hành vi “ôm chặt ví” hàm ý họ dễ trộm cắp; dấu nháy “kinh nghiệm xương máu” mang sắc thái mỉa mai/hợp thức hóa định kiến; không dùng từ chửi thẳng.
Bước 3: Xúc phạm gián tiếp: hàm ý quy chụp tật xấu cho cả nhóm qua gợi ý, mỉa mai và khái quát hoá.
Bước 4: Mục đích củng cố nghi ngờ mặc định đối với một cộng đồng, gieo sợ hãi và khuyến khích né tránh/kỳ thị dù không kêu gọi công khai.
label: Implicit HS
Tổng hợp: Bước 1 xác định đối tượng là “người nói tiếng [ngôn ngữ Y]” (nhóm được bảo vệ). Bước 2 chỉ ra cấu trúc khái quát hóa “cứ thấy… là…”, hành vi “ôm chặt ví” hàm ý họ dễ trộm cắp và dấu nháy “kinh nghiệm xương máu” mang sắc thái mỉa mai; không có từ chửi thẳng. Bước 3 kết luận là xúc phạm gián tiếp. Bước 4 cho thấy mục đích củng cố nghi ngờ, gieo sợ hãi và khuyến khích né tránh/kỳ thị dù không kêu gọi công khai — do đó phân loại là Implicit HS.

văn bản 2: "Tôi phản đối việc trường đại học ưu tiên ứng viên có điểm phỏng vấn cao hơn điểm học thuật; tiêu chí nên minh bạch và dựa trên số liệu."
Bước 1: Không nhắm vào cá nhân hay nhóm được bảo vệ; đối tượng là một chính sách tuyển sinh.
Bước 2: Ngôn từ trung tính, lập luận (“phản đối”, “nên”, “minh bạch”, “dựa trên số liệu”) không có từ miệt thị, sỉ nhục hay gán nhãn.
Bước 3: Không có hình thức xúc phạm nào.
Bước 4: Mục đích là phản biện chính sách và đề xuất cải thiện; không nhằm làm tổn thương hay lan truyền định kiến.
label: Non HS
Tổng hợp: Bước 1 xác định không nhắm cá nhân hay nhóm được bảo vệ, mà là một chính sách tuyển sinh. Bước 2 ghi nhận ngôn từ trung tính, lập luận (“phản đối”, “nên”, “minh bạch”, “dựa trên số liệu”) và không có từ miệt thị. Bước 3 kết luận không có hình thức xúc phạm nào. Bước 4 cho thấy mục đích là phản biện chính sách và đề xuất cải thiện, không nhằm gây tổn hại hay lan truyền định kiến — do đó phân loại là Non HS.

văn bản 3: Bọn người Hồi giáo theo đạo Hồi là lũ khủng bố, cấm họ bén mảng tới khu này.”
Bước 1: Nhắm tới “người theo đạo Hồi” — một nhóm tôn giáo (nhóm được bảo vệ).
Bước 2: Từ ngữ “bọn”, “lũ” có sắc thái phi nhân hoá; “bọn người Hồi giáo” là từ xúc phạm trực diện; “khủng bố” là gán nhãn tội phạm; mệnh lệnh “cấm họ…” thúc đẩy loại trừ.
Bước 3: Xúc phạm trực tiếp: dùng từ ngữ mang tính miệt thị rõ ràng và kêu gọi loại trừ.
Bước 4: Mục đích thể hiện sự hạ nhục và gắn định kiến nguy hiểm với một tôn giáo; có khả năng kích động kỳ thị và hành vi phân biệt đối xử.
label: Explicit HS
Tổng hợp: Bước 1 xác định đối tượng là “người theo đạo Hồi” (nhóm được bảo vệ). Bước 2 ghi nhận các từ ngữ “bọn”, “lũ”, “người theo đạo Hồi”, gán nhãn “khủng bố” và mệnh lệnh “cấm họ…”, cho thấy ngôn từ miệt thị và kêu gọi loại trừ. Bước 3 kết luận đây là xúc phạm trực tiếp. Bước 4 nêu mục đích hạ nhục, gắn định kiến nguy hiểm và có khả năng kích động kỳ thị/phân biệt — do đó phân loại là Explicit HS.
"""


# ------------------------------------------------------------------ ZERO-SHOT 
def make_zero_prompt(model_name: str):
    m = model_name.lower()

    if m.startswith(("llama-2", "llama_2", "mistral")):
        return "<s>[INST] " + _VI_INSTRUCTION + "\n\nVăn bản: {text} [/INST]"

    if m.startswith(("llama-3", "llama_3")):
        return (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _VI_INSTRUCTION +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )

    if m.startswith("qwen"):
        return (
            "<|im_start|>system\n" + _VI_INSTRUCTION + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )

    if m.startswith("gemma"):
        return f"system: {_VI_INSTRUCTION}\nuser: Văn bản: {{text}}\nassistant:"

    if m in {"gpt-4o", "gpt-4", "gpt-3.5-turbo"}:
        return [
            {"role": "system", "content": _VI_INSTRUCTION},
            {"role": "user",  "content": "Văn bản: {text}"},
        ]

    raise ValueError(f"[make_zero_prompt] unsupported model: {model_name}")

# ------------------------------------------------------------------ FEW-SHOT
def _render_examples(template: str) -> str:
    """Biến list `_FEWSHOT_EXAMPLES` thành chuỗi đã format."""
    return "".join(template.format(**ex) for ex in _FEWSHOT_EXAMPLES)

def make_fewshot_prompt_with_available_exp(model_name: str):
    """
    Trả về dict với 3 khóa: 'examples' (empty), 'example_tpl' (không dùng),
    'suffix' (chứa sẵn instruction + examples + đoạn user). 
    """
    m = model_name.lower()

    # ---------------- Llama-2 / Mistral ----------------
    if m.startswith(("llama-2", "llama_2", "mistral")):
        rendered = _render_examples(_EX_TPL_GENERIC)
        suffix = (
            "<s>[INST] " + _VI_INSTRUCTION + " [/INST]\n"  # prefix
            + rendered +                                  # examples
            "<s>[INST] " + _SUFFIX_USER_GENERIC + " [/INST]"  # user
        )
        return {"examples": [], "example_tpl": _EX_TPL_GENERIC, "suffix": suffix}

    # ---------------- Llama-3 --------------------------
    if m.startswith(("llama-3", "llama_3")):
        rendered = _render_examples(_EX_TPL_LLAMA3)
        suffix = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _VI_INSTRUCTION +
            "\n<|end_header_id|>\n"
            + rendered +
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        return {"examples": [], "example_tpl": _EX_TPL_LLAMA3, "suffix": suffix}

    # ---------------- Qwen -----------------------------
    if m.startswith("qwen"):
        rendered = _render_examples(_EX_TPL_GENERIC)
        suffix = (
            "<|im_start|>system\n" + _VI_INSTRUCTION + "<|im_end|>\n"
            + rendered +
            "<|im_start|>user\n" + _SUFFIX_USER_GENERIC + "<|im_end|>\n"
            "<|im_start|>assistant"
        )
        return {"examples": [], "example_tpl": _EX_TPL_GENERIC, "suffix": suffix}

    # ---------------- Gemma ----------------------------
    if m.startswith("gemma"):
        rendered = _render_examples(_EX_TPL_GENERIC)
        suffix = (
            f"system: {_VI_INSTRUCTION}\n"
            + rendered +
            f"user: {_SUFFIX_USER_GENERIC}\nassistant:"
        )
        return {"examples": [], "example_tpl": _EX_TPL_GENERIC, "suffix": suffix}

    raise ValueError(f"[make_fewshot_prompt] unsupported model: {model_name}")

# ------------------------------------------------------------------ self-generative FEW-SHOT 

def generate_few_shot_examples(agent: ChatAgent, model_name: str, n: int = 5) -> List[Dict[str, str]]:
    instruction = _GENERATION_INSTRUCTION.format(n=n)
    # print("tôi là generate_few_shot_examples")
    # Format prompt phù hợp từng loại model
    if model_name.startswith(("llama-3","llama_3")):
        prompt = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + instruction +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nTạo ví dụ.\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
    elif model_name.startswith(("llama-2", "llama_2", "mistral")):
        prompt = f"<s>[INST] {instruction} [/INST]"
    elif model_name.startswith("qwen"):
        prompt = (
            "<|im_start|>system\n" + instruction + "<|im_end|>\n"
            "<|im_start|>user\nTạo ví dụ.<|im_end|>\n"
            "<|im_start|>assistant"
        )
    elif model_name.startswith("gemma"):
        prompt = f"system: {instruction}\nuser: Tạo ví dụ.\nassistant:"
    else:
        raise ValueError(f"[generate_few_shot_examples] unsupported model: {model_name}")
    
    # Gọi inference
    response_text = agent.inference(prompt_template=prompt, input_values={})
    print('response_text: ',response_text)
    # Parse kết quả thành danh sách ví dụ
    examples = []

    # if model_name.startswith("mistral"):
    #     import re
    #     # Regex: bắt "Văn bản 1: ..." (có thể xuống dòng), label ở sau, có thể dính liền
    #     pattern = r'Văn bản\s*\d+\s*:\s*\n?["“]?(.*?)["”]?\s*\n*label\s*:\s*(Explicit HS|Implicit HS|Non HS)'
    #     matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)

    #     for text, label in matches:
    #         cleaned_text = text.strip()
    #         cleaned_label = label.strip()
    #         if cleaned_text and cleaned_label:
    #             examples.append({
    #                 "text": cleaned_text,
    #                 "label": cleaned_label
    #             })
    # elif model_name.startswith(("llama-3", "llama_3")):
    #     # Bắt đoạn dạng:
    #     # <|start_header_id|>user\nVăn bản: ...<|end_header_id|>
    #     # <|start_header_id|>assistant\n<label><|end_header_id|>
    #     pattern = re.compile(
    #         r"<\|start_header_id\|>user\s*Văn bản:\s*(.*?)<\|end_header_id\|>\s*"
    #         r"<\|start_header_id\|>assistant\s*(Explicit HS|Implicit HS|Non HS)\s*<\|end_header_id\|>",
    #         re.DOTALL
    #     )
    #     matches = pattern.findall(response_text)
    #     for text, label in matches:
    #         examples.append({
    #             "text": text.strip(),
    #             "label": label.strip()
    #         })
    # else:
    #     for block in response_text.strip().split("\n\n"):
    #         lines = block.strip().split("\n")
    #         if len(lines) == 2 and lines[0].startswith("Văn bản:") and lines[1].startswith("label:"):
    #             text = lines[0].replace("Văn bản:", "").strip()
    #             label = lines[1].replace("label:", "").strip()
    #             examples.append({"text": text, "label": label})
    import re
    if model_name.startswith(("llama-3", "llama_3")):
        pattern = re.compile(
            r"<\|start_header_id\|>user\s*Văn bản:\s*(.*?)<\|end_header_id\|>\s*"
            r"<\|start_header_id\|>assistant\s*(Explicit HS|Implicit HS|Non HS)\s*<\|end_header_id\|>",
            re.DOTALL
        )
        matches = pattern.findall(response_text)
        for text, label in matches:
            examples.append({"text": text.strip(), "label": label.strip()})

    # ======== Case 2: Mistral-style numbered "Văn bản X:" format =========
    pattern_vanban = re.compile(
        r'Văn bản\s*\d+\s*:\s*["“]?(.*?)["”]?\s*label\s*:\s*(Explicit HS|Implicit HS|Non HS)',
        re.DOTALL | re.IGNORECASE
    )
    for text, label in pattern_vanban.findall(response_text):
        examples.append({"text": text.strip(), "label": label.strip()})

    # ======== Case 3: Simple numbered list like `1. "..." label: ...` =========
    pattern_numbered_list = re.compile(
        r'\d+\.\s*["“]?(.*?)["”]?\s*label\s*:\s*(Explicit HS|Implicit HS|Non HS)',
        re.DOTALL | re.IGNORECASE
    )
    for text, label in pattern_numbered_list.findall(response_text):
        examples.append({"text": text.strip(), "label": label.strip()})

    # ======== Case 4: Markdown-style "**Văn bản:** ..." \n "**label:** ..." =========
    pattern_markdown = re.compile(
        r'\d+\.\s*\*\*Văn bản:\*\*\s*["“]?(.*?)["”]?\s*\n\s*\*\*label:\*\*\s*(Explicit HS|Implicit HS|Non HS)',
        re.DOTALL | re.IGNORECASE
    )
    for text, label in pattern_markdown.findall(response_text):
        examples.append({"text": text.strip(), "label": label.strip()})


    # ======== Case 4: Fallback "Văn bản: ... \n label: ..." =========
    if not examples:
        for block in response_text.strip().split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) == 2 and lines[0].startswith("Văn bản:") and lines[1].startswith("label:"):
                text = lines[0].replace("Văn bản:", "").strip()
                label = lines[1].replace("label:", "").strip()
                examples.append({"text": text, "label": label})
        
    return examples,response_text

def make_prompt_for_self_generative_fewshot(model_name: str, examples: List[Dict[str, str]]) -> Dict:
    m = model_name.lower()
    
    if m.startswith(("llama-2", "llama_2", "mistral")):
        tpl = _EX_TPL_GENERIC
        rendered = "".join(tpl.format(**ex) for ex in examples)
        suffix = (
            "<s>[INST] " + _VI_INSTRUCTION + " [/INST]\n"
            + rendered +
            "<s>[INST] " + _SUFFIX_USER_GENERIC + " [/INST]"
        )
        return {"examples": [], "example_tpl": tpl, "suffix": suffix}

    elif m.startswith(("llama-3","llama_3")):
        tpl = _EX_TPL_LLAMA3
        rendered = "".join(tpl.format(**ex) for ex in examples)
        suffix = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _VI_INSTRUCTION +
            "\n<|end_header_id|>\n"
            + rendered +
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        return {"examples": [], "example_tpl": tpl, "suffix": suffix}
    
    elif m.startswith("qwen"):
        tpl = _EX_TPL_GENERIC
        rendered = "".join(tpl.format(**ex) for ex in examples)
        suffix = (
            "<|im_start|>system\n" + _VI_INSTRUCTION + "<|im_end|>\n"
            + rendered +
            "<|im_start|>user\n" + _SUFFIX_USER_GENERIC + "<|im_end|>\n"
            "<|im_start|>assistant"
        )
        return {"examples": [], "example_tpl": tpl, "suffix": suffix}

    elif m.startswith("gemma"):
        tpl = _EX_TPL_GENERIC
        rendered = "".join(tpl.format(**ex) for ex in examples)
        suffix = (
            f"system: {_VI_INSTRUCTION}\n"
            + rendered +
            f"user: {_SUFFIX_USER_GENERIC}\nassistant:"
        )
        return {"examples": [], "example_tpl": tpl, "suffix": suffix}

    raise ValueError(f"[make_dynamic_fewshot_prompt] unsupported model: {model_name}")



#-------------------------------------make CoT prompt-------------------
def make_CoT_prompt(model_name: str):
    m = model_name.lower()

    if m.startswith(("llama-2", "llama_2", "mistral")):
        return "<s>[INST] " + _SIMPLE_CoT_INSTRUCTION + "\n\nVăn bản: {text} [/INST]"

    if m.startswith(("llama-3", "llama_3")):
        return (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _SIMPLE_CoT_INSTRUCTION_SCEN1 +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )

    if m.startswith("qwen"):
        return (
            "<|im_start|>system\n" + _SIMPLE_CoT_INSTRUCTION + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )

    if m.startswith("gemma"):
        return f"system: {_SIMPLE_CoT_INSTRUCTION}\nuser: Văn bản: {{text}}\nassistant:"

    raise ValueError(f"[make_zero_prompt] unsupported model: {model_name}")

def make_CoT_prompt_scen2(model_name: str):
    m = model_name.lower()

    if m.startswith(("llama-2", "llama_2", "mistral")):
        return "<s>[INST] " + _SIMPLE_CoT_INSTRUCTION_SCEN2 + "\n\nVăn bản: {text} [/INST]"

    if m.startswith(("llama-3", "llama_3")):
        return (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _SIMPLE_CoT_INSTRUCTION_SCEN2 +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )

    if m.startswith("qwen"):
        return (
            "<|im_start|>system\n" + _SIMPLE_CoT_INSTRUCTION_SCEN2 + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )

    if m.startswith("gemma"):
        return f"system: {_SIMPLE_CoT_INSTRUCTION_SCEN2}\nuser: Văn bản: {{text}}\nassistant:"

    raise ValueError(f"[make_zero_prompt] unsupported model: {model_name}")

def make_CoT_prompt_scen3(model_name: str):
    m = model_name.lower()

    if m.startswith(("llama-2", "llama_2", "mistral")):
        return "<s>[INST] " + _SIMPLE_CoT_INSTRUCTION_SCEN3 + "\n\nVăn bản: {text} [/INST]"

    if m.startswith(("llama-3", "llama_3")):
        return (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _SIMPLE_CoT_INSTRUCTION_SCEN3 +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )

    if m.startswith("qwen"):
        return (
            "<|im_start|>system\n" + _SIMPLE_CoT_INSTRUCTION_SCEN3 + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )

    if m.startswith("gemma"):
        return f"system: {_SIMPLE_CoT_INSTRUCTION_SCEN3}\nuser: Văn bản: {{text}}\nassistant:"

    raise ValueError(f"[make_zero_prompt] unsupported model: {model_name}")


def make_CoT_prompt_scen1(model_name: str):
    m = model_name.lower()

    if m.startswith(("llama-2", "llama_2", "mistral")):
        return "<s>[INST] " + _SIMPLE_CoT_INSTRUCTION_SCEN1 + "\n\nVăn bản: {text} [/INST]"

    if m.startswith(("llama-3", "llama_3")):
        return (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _SIMPLE_CoT_INSTRUCTION_SCEN1 +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )

    if m.startswith("qwen"):
        return (
            "<|im_start|>system\n" + _SIMPLE_CoT_INSTRUCTION_SCEN1 + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )

    if m.startswith("gemma"):
        return f"system: {_SIMPLE_CoT_INSTRUCTION_SCEN1}\nuser: Văn bản: {{text}}\nassistant:"

    raise ValueError(f"[make_zero_prompt] unsupported model: {model_name}")

def make_CoT_two_prompts(model_name: str):

    m = model_name.lower()

    # Template cho bước 1 (phân tích từng bước)
    context_template = _COT2_CONTEXT + "\n\nVăn bản: {text}"
    # Template cho bước 2 (phân loại nhãn)
    label_template = _COT2_LABEL

    if m.startswith(("llama-2", "llama_2", "mistral")):
        context_template = "<s>[INST] " + context_template + " [/INST]"
        label_template = "<s>[INST] " + label_template + "\n\nVăn bản: {text} [/INST]"
        return context_template, label_template

    if m.startswith(("llama-3", "llama_3")):
        context_template = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _COT2_CONTEXT +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        label_template = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _COT2_LABEL +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        return context_template, label_template

    if m.startswith("qwen"):
        context_template = (
            "<|im_start|>system\n" + _COT2_CONTEXT + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )
        label_template = (
            "<|im_start|>system\n" + _COT2_LABEL + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )
        return context_template, label_template

    if m.startswith("gemma"):
        context_template = f"system: {_COT2_CONTEXT}\nuser: Văn bản: {{text}}\nassistant:"
        label_template = f"system: {_COT2_LABEL}\nuser: Văn bản: {{text}}\nassistant:"
        return context_template, label_template

    raise ValueError(f"[make_CoT_two_prompts] unsupported model: {model_name}")



def make_CoT_two_prompts_random_fewshot(model_name: str):

    m = model_name.lower()

    # Template cho bước 1 (phân tích từng bước)
    context_template = _COT2_CONTEXT + "\n\nVăn bản: {text}"
    # Template cho bước 2 (phân loại nhãn)
    label_template = _COT2_LABEL + _COT2_LABEL_FEWSHOT

    if m.startswith(("llama-2", "llama_2", "mistral")):
        context_template = "<s>[INST] " + context_template + " [/INST]"
        label_template = "<s>[INST] " + label_template + "\n\nVăn bản: {text} [/INST]"
        return context_template, label_template

    if m.startswith(("llama-3", "llama_3")):
        context_template = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _COT2_CONTEXT +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        label_template = (
            "<|begin_of_text|><|start_header_id|>system\n"
            + _COT2_LABEL +
            "\n<|end_header_id|>\n"
            "<|start_header_id|>user\nVăn bản: {text}\n<|end_header_id|>\n"
            "<|start_header_id|>assistant\n"
        )
        return context_template, label_template

    if m.startswith("qwen"):
        context_template = (
            "<|im_start|>system\n" + _COT2_CONTEXT + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )
        label_template = (
            "<|im_start|>system\n" + _COT2_LABEL + "<|im_end|>\n"
            "<|im_start|>user\nVăn bản: {text}<|im_end|>\n"
            "<|im_start|>assistant"
        )
        return context_template, label_template

    if m.startswith("gemma"):
        context_template = f"system: {_COT2_CONTEXT}\nuser: Văn bản: {{text}}\nassistant:"
        label_template = f"system: {_COT2_LABEL}\nuser: Văn bản: {{text}}\nassistant:"
        return context_template, label_template

    raise ValueError(f"[make_CoT_two_prompts] unsupported model: {model_name}")




