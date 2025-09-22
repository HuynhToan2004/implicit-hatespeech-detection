from transformers import AutoModelForCausalLM, AutoTokenizer
# import itertools
import pandas as pd
import json
from tqdm import tqdm
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
model_name = "/data2/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/Qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def call_llm(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=True,        # bật sampling
        temperature=0.9,       # điều chỉnh độ "sáng tạo" (0.7–1.0 hợp lý)
        top_p=0.95,            # nucleus sampling (giữ lại 95% xác suất)
        # top_k=50               # (tùy chọn) chỉ lấy 50 token có xác suất cao nhất
    )

    # Cắt phần input ra, chỉ giữ phần model sinh mới
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

_GENERATION_INSTRUCTION = """
Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. 
Hãy tạo ra {n} ví dụ phức tạp bằng ## TIẾNG VIỆT phân loại phát ngôn thành 1 trong 3 nhãn sau:
- Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
- Implicit HS –  là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt mà đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
- Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.

Các ví dụ phải đa dạng các chủ đề, tấn công vào nhiều mặt khác nhau của các thành phần trong xã hội, không nhất thiết là chỉ ở Việt Nam.
Mỗi ví dụ chỉ gồm 1 văn bản và một nhãn theo định dạng:
Văn bản: <văn bản>
label: <label>
Không được giải thích thêm.
"""
    
def generate_samples(n):
    system_prompt = _GENERATION_INSTRUCTION.format(n=n)
    user_prompt = "Hãy tạo ra {} ví dụ như trên.".format(n)
    response = call_llm(system_prompt, user_prompt)

    samples = []
    for block in response.split("Văn bản: ")[1:]:
        try:
            text, label = block.split("label: ")
            text = text.strip().replace("\n", " ")
            label = label.strip().replace("\n", " ")
            samples.append({"text": text, "label": label})
        except:
            continue
    return samples

# ---- RUN & SAVE JSONL ----
if __name__ == "__main__":
    print(device)
    out_file = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/generated_data5.jsonl"

    with open(out_file, "a", encoding="utf-8") as f:  
        for i in tqdm(range(800), desc="Generating"):
            batch_samples = generate_samples(50)  # mỗi lần 50 ví dụ
            for sample in batch_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("✅ Hoàn tất: sinh ~40,000 samples và lưu vào", out_file)

