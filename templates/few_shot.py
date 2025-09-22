

FEWSHOT_PROMPT = """[INS]
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn. Phân loại câu sau vào một trong ba nhãn sau:
- Non HS: Không phải là phát ngôn thù ghét.
- Explicit HS: Phát ngôn thù ghét rõ ràng, công kích trực tiếp hoặc xúc phạm cá nhân, nhóm đối tượng.
- Implicit HS: Phát ngôn thù ghét ẩn ý, ám chỉ, mỉa mai, không công kích trực tiếp.
Chỉ trả về một nhãn duy nhất: Non HS, Explicit HS hoặc Implicit HS. Không chấp nhận các câu trả lời khác. 

### Đây là các ví dụ:
- Ví dụ 1: "Người da trắng, người da đen, hay người châu Á, ai cũng có thể phạm sai lầm." Nhãn: "Non HS".
- Ví dụ 2: "Mấy đứa chuyển giới cũng là con người, nhưng có lẽ xã hội chưa sẵn sàng để hiểu tụi nó" Nhãn: "Non HS".
- Ví dụ 3: "Nhiều người không ủng hộ việc đàn áp người di cư bất hợp pháp bằng bạo lực." Nhãn: "Non HS".
- Ví dụ 4: "Đừng để bọn đồng tính dạy dỗ con cái chúng ta." Nhãn: "Explicit HS".
- Ví dụ 5:  "Tụi người da đen toàn là lũ vô học và nguy hiểm." Nhãn: "Explicit HS".
- Ví dụ 6:  "Bọn chuyển giới thật kinh tởm, không thể nào là người bình thường được." Nhãn: "Explicit HS".
- Ví dụ 7: "Không hiểu sao dạo này công ty toàn tuyển người khuyết tật vào làm ở quầy lễ tân nhỉ." Nhãn: "Implicit HS".
- Ví dụ 8: "Thật vui khi thấy một cô gái làm lập trình – điều đó vẫn hiếm và hơi... bất ngờ." Nhãn: "Implicit HS".
- Ví dụ 9: "Người chuyển giới có thể sống như mình muốn, miễn là đừng bắt người khác gọi họ bằng đại từ kỳ lạ." Nhãn: "Implicit HS".
- Ví dụ 10: ""Đâu phải ai ghét người đồng tính, chỉ là nhiều người không muốn con mình bị ảnh hưởng."" Nhãn: "Implicit HS".
[/INS]
[INPUT]
Văn bản: {sentence}
[OUTPUT]
label:
"""