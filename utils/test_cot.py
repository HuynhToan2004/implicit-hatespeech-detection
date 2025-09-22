# from transformers import AutoModelForCausalLM, AutoTokenizer
# import itertools
# import pandas as pd
# from tqdm import tqdm
# import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# model_name = "/data2/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/Qwen2.5-14B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# def call_llm(system_prompt, user_prompt):
#     messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512
#     )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response


# # Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn trên tiếng Việt. Với định nghĩa các loại phát ngôn:
# # - Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
# # - Implicit HS – là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt, thường sử dụng từ ngữ miệt thị gián tiếp, hoặc sẽ đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
# # - Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.

# SYS_PROMPT_COT2_1 = """

# Hãy phân tích câu phát biểu theo các bước sau và chỉ trả về kết quả phân tích, *không đưa ra nhãn cuối cùng*:
#     - Bước 1: Xác định xem có cá nhân hoặc nhóm đối tượng nào trong câu nói không.
#     - Bước 2: Phân tích từ ngữ và cách diễn đạt trong câu phát biểu.
#     - Bước 3: Xác định hình thức xúc phạm của câu, trực tiếp hay gián tiếp.
#     - Bước 4: Phân tích mục đích: Câu nói nhằm mục đích gì? Có nhằm làm tổn thương, phỉ báng, lan truyền định kiến hoặc kích động thù ghét, v.v hay không?

#     *Định dạng trả về*:
#     - Bước 1: <kết quả phân tích đối tượng>
#     - Bước 2: <kết quả phân tích từ ngữ>
#     - Bước 3: <kết quả phân hình thức của câu>
#     - Bước 4: <kết quả phân tích mục đích của câu>
# """

# # Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn trên tiếng Việt.
# # Dựa trên phân tích được cung cấp, hãy phân loại câu sau vào **một trong ba nhãn sau**:
# # - Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
# # - Implicit HS – là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt, thường sử dụng từ ngữ miệt thị gián tiếp, hoặc sẽ đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
# # - Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.


# SYS_PROMPT_COT2_2 = """

# Phân tích được cung cấp:
# {analysis}

# Chỉ dựa trên các phân tích trên, *Không tự ý suy luận thêm*, hãy:
# 1. Tổng hợp các yếu tố từ các bước, đảm bảo tham chiếu rõ ràng đến từng bước.
# 2. Phân loại nhãn cho câu phát biểu (Explicit HS, Implicit HS, hoặc Non HS).

# *Định dạng trả về*:
# - label: <label>
# - Tổng hợp: <tóm tắt các yếu tố từ các bước phân tích, nêu rõ cách sử dụng từng bước để đưa ra nhãn>
# """


# # SYS_PROMPT_COT2_1 = """
# # Bạn là một chuyên gia ngôn ngữ phân tích phát ngôn tiếng Việt. Hãy phân tích theo định nghĩa:

# # - Explicit HS: xúc phạm/thù ghét trực diện (từ miệt thị nghĩa đen, dehumanize, mệnh lệnh/đe doạ, kêu gọi loại trừ/tước quyền) nhắm vào cá nhân/nhóm.
# # - Implicit HS: thù ghét ẩn/gián tiếp (mỉa mai, ẩn dụ, phóng đại, câu hỏi tu từ, “không … nhưng …”, khái quát định kiến).
# # - Non HS: trung lập/tích cực/phản biện hợp lý; không xúc phạm, không phân biệt đối xử.

# # Các bước phân tích (KHÔNG đưa ra nhãn cuối cùng):

# # Bước 1 — Đối tượng:
# # - Liệt kê đối tượng bị nhắc tới (cá nhân/nhóm). Đánh dấu: {protected? yes/no}. 
# #   Ví dụ protected: sắc tộc/dân tộc, tôn giáo, giới, xu hướng tính dục, quốc tịch/vùng miền, khuyết tật, nhập cư, v.v.
# # - Nếu chỉ nhắm vào ý tưởng/chính sách/tổ chức (không đánh vào căn tính): ghi rõ.

# # Bước 2 — Ngôn ngữ & chỉ báo bề mặt:
# # - Trích các cụm từ/đoạn NGẮN làm bằng chứng (có thể che bớt: “đ**n”, “[SLUR]”).
# # - Phân loại từng chỉ báo: {slur/insult, khái quát (“bọn/tụi/lũ…”, “mọi [nhóm]…”), mệnh lệnh/đe doạ/tước quyền, phủ định giả “không … nhưng …”, mỉa mai/ẩn dụ, xúc phạm năng lực/phẩm chất}.
# # - Đánh dấu polarity: {tiêu cực / trung tính / mô tả trích dẫn}.

# # Bước 3 — Hình thức (directness):
# # - Kết luận: {Direct / Indirect / None}.
# # - Tiêu chí:
# #   • Direct nếu có slur/insult nghĩa đen, mệnh lệnh/đe doạ/tước quyền, dehumanize, quy kết ác ý thẳng vào đối tượng.
# #   • Indirect nếu biểu hiện qua mỉa mai, ẩn dụ, phóng đại, “không … nhưng …”, khái quát định kiến, câu hỏi tu từ.
# #   • None nếu không đủ chỉ báo.

# # Bước 4 — Mục đích/ý đồ (nếu suy ra được từ chính câu):
# # - Phân loại: {Harm/Derogation, Exclusion/Deprivation, Incitement, Propaganda/Stereotype, None}.
# # - Dẫn chứng ngắn gọn (trích cụm).

# # Ràng buộc & lưu ý:
# # - Phân biệt “người [quốc tịch]” (target protected) với “chính phủ/chính sách [quốc gia]” (không đánh vào căn tính).
# # - Lời trích dẫn/báo cáo lại (reporting) ≠ tán thành: ghi rõ “trích dẫn” trong Bước 2 nếu có dấu hiệu.
# # - Không suy diễn ngoài văn bản; chỉ dùng thông tin thấy được.
# # - Chỉ trả về theo định dạng dưới đây.

# # Định dạng trả về:
# # - Bước 1: <đối tượng; protected? yes/no; cá nhân/nhóm; hay ý tưởng/chính sách/tổ chức?>
# # - Bước 2: <các chỉ báo + trích dẫn ngắn + phân loại + polarity>
# # - Bước 3: <Direct/Indirect/None + lý do ngắn dựa trên trích dẫn ở Bước 2>
# # - Bước 4: <loại ý đồ + bằng chứng ngắn>

# # """

# # SYS_PROMPT_COT2_2 = """
# # Bạn là một chuyên gia ngôn ngữ. Dựa DUY NHẤT trên “Phân tích được cung cấp” (không suy luận thêm ngoài nội dung đó), 
# # Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các câu phát ngôn trên tiếng Việt. Với định nghĩa các loại phát ngôn:
# # - Explicit HS – là phát ngôn thể hiện sự thù ghét, xúc phạm một cách trực tiếp để nhắm đến các nhóm đối tượng hoặc một cá nhân, thường sử dụng từ ngữ mang nghĩa đen có tính miệt thị trực tiếp cao như các từ ngữ phân biệt chủng tộc, kỳ thị giới tính hoặc xúc phạm tôn giáo. Các biểu hiện xúc phạm, kì thị trong câu có thể nhận diện rõ ràng qua các từ ngữ trong câu mà không cần phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội.
# # - Implicit HS – là phát ngôn thù ghét ẩn dụ hoặc gián tiếp, sử dụng từ ngữ mỉa mai, châm biếm, ẩn dụ, câu hỏi tu từ, hay phóng đại nhằm lồng ghép định kiến tiêu cực đối với một cá nhân hoặc nhóm đối tượng. Các biểu hiện implicit thường không thể phát hiện qua từ ngữ bề mặt, thường sử dụng từ ngữ miệt thị gián tiếp, hoặc sẽ đòi hỏi phải suy luận ý đồ, ngữ cảnh và kiến thức xã hội để hiểu được thông điệp kỳ thị được ngụy trang bên dưới.
# # - Non HS – là phát ngôn không mang nội dung kỳ thị, xúc phạm hay phân biệt đối xử. Câu mang tính trung lập, tích cực, phản biện hợp lý hoặc thể hiện thông tin không gây tổn hại đến bất kỳ cá nhân hay nhóm xã hội nào.

# # Phân tích được cung cấp:
# # {analysis}

# # QUY TẮC RA QUYẾT ĐỊNH (tham chiếu từng bước): 
# # 1) Nếu Bước 1 cho thấy KHÔNG có cá nhân/nhóm bị nhắm tới **hoặc** chỉ nhắm vào ý tưởng/chính sách/tổ chức và Bước 2–3 không có chỉ báo xúc phạm hướng vào căn tính ⇒ **Non HS**.
# # 2) Nếu Bước 2 có bất kỳ chỉ báo trực diện mạnh (slur nghĩa đen, mệnh lệnh/đe doạ/tước quyền, dehumanize) **hoặc** Bước 3 = Direct ⇒ **Explicit HS**.
# # 3) Nếu (2) không thoả, nhưng Bước 3 = Indirect **và** Bước 2 có ≥1 chỉ báo ẩn ý đáng kể (mỉa mai/ẩn dụ; “không … nhưng …”; khái quát định kiến; câu hỏi tu từ quy chụp tiêu cực) ⇒ **Implicit HS**.
# # 4) Nếu không đủ điều kiện (2) hoặc (3) ⇒ **Non HS**.

# # TIE-BREAK (giảm nhầm Explicit↔Implicit & over-flag):
# # - Khi đồng thời có dấu hiệu Direct và Indirect ⇒ **chọn Explicit HS**.
# # - Khi phân vân giữa Implicit và Non HS: CHỈ gán **Implicit** nếu có ít nhất một trong các bằng chứng sau từ Bước 2–3:
# #   (a) “không … nhưng …” kèm đánh giá tiêu cực về nhóm/cá nhân; 
# #   (b) khái quát tiêu cực (“bọn/tụi/lũ…”, “mọi [nhóm] đều …”);
# #   (c) mỉa mai/ẩn dụ làm suy giảm phẩm giá nhóm/cá nhân.
# #   Nếu không có, ⇒ **Non HS**.
# # - Nationalities/Regions: phân biệt “người [quốc tịch/vùng]” (protected) với “quốc gia/chính phủ/chính sách” (không đánh vào căn tính). Chỉ gán HS khi sự tiêu cực nhắm vào **con người** của nhóm đó.

# # Định dạng trả về:
# # - label: <Explicit HS | Implicit HS | Non HS>
# # - Tổng hợp: <tóm tắt cách bạn áp dụng Bước 1–4 để ra nhãn, nêu rõ chỉ báo then chốt và luật/tie-break được dùng>
# # """


# # ==== Pipeline ====
# def pipeline(sample_text):
#     # Step 1: Prompt 1 → reasoning
#     reasoning_full = call_llm(SYS_PROMPT_COT2_1, sample_text)

#     # Parse thành dict {1:..., 2:..., 3:..., 4:...}
#     reasoning_steps = {}
#     for i in range(1, 5):
#         marker = f"Bước {i}:"
#         try:
#             if i < 4:
#                 part = reasoning_full.split(marker, 1)[1].split(f"Bước {i + 1}:", 1)[0].strip()
#             else:
#                 part = reasoning_full.split(marker, 1)[1].strip()
#             reasoning_steps[i] = f"{marker} {part}"
#         except Exception as e:
#             reasoning_steps[i] = f"{marker} (Không parse được - {str(e)})"

#     print("\n[Reasoning Full]")
#     for k, v in reasoning_steps.items():
#         print(v)

#     # Step 2: Sinh các biến thể reasoning
#     step_ids = [1, 2, 3, 4]
#     all_variants = []
#     for r in range(1, len(step_ids) + 1):
#         for combo in itertools.combinations(step_ids, r):
#             combined = "\n".join([reasoning_steps[i] for i in combo])
#             all_variants.append((combo, combined))

#     # Thêm full reasoning
#     all_variants.append((tuple(step_ids), "\n".join([reasoning_steps[i] for i in step_ids])))


#     output_dict = {}
#     for combo, analysis in all_variants:
#         prompt2_filled = SYS_PROMPT_COT2_2.format(analysis=analysis)
#         label_output = call_llm("", prompt2_filled)

#         combo_name = "_".join(map(str, combo))
#         output_dict[f"steps_{combo_name}_reasoning"] = analysis
#         output_dict[f"steps_{combo_name}_output"] = label_output  # Lưu nguyên văn

#     return output_dict


# if __name__ == "__main__":
#     TEXT_COLUMN = "translation"
#     input_path = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/300_mau.csv"
#     output_path = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/cot2_detail/qwen_2.5_14B_new_prompt_no_sample_no_instruct.csv"

#     df = pd.read_csv(input_path, encoding="utf-8")

#     # Tạo file output rỗng (chỉ có header) trước
#     first_row = True  

#     for i, row in tqdm(df.iterrows(), total=len(df)):
#         text = row[TEXT_COLUMN]
#         res_dict = pipeline(text)
#         res_dict[TEXT_COLUMN] = text

#         # Ghép kết quả với thông tin gốc
#         result_row = pd.concat([row.to_frame().T.reset_index(drop=True), 
#                                 pd.DataFrame([res_dict])], axis=1)

#         # Ghi thẳng vào file sau mỗi vòng lặp
#         result_row.to_csv(output_path, 
#                         mode="a", 
#                         header=first_row, 
#                         index=False, 
#                         encoding="utf-8")
        
#         first_row = False

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import re
import matplotlib.pyplot as plt

VALID_LABELS = ["Non HS", "Explicit HS", "Implicit HS"]

def extract_label(text):
    # Tìm nhãn hợp lệ trong text
    match = re.search(r'\b(Non HS|Implicit HS|Explicit HS)\b', str(text))
    if match:
        return match.group(1)
    return None   # trả về None nếu không tìm thấy

def extract_label_from_text(text):
    """
    Trích xuất nhãn từ chuỗi dạng:
    - label: Implicit HS
    - Tổng hợp: ...
    """
    if pd.isna(text):
        return None
    match = re.search(r"label:\s*(.+)", str(text))
    if match:
        label = match.group(1).strip()
        if label in VALID_LABELS:
            return label
    return None


# ========== Load dữ liệu ========== 
df_2_prompts = pd.read_csv(
    "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/cot2_detail/qwen_2.5_14B_new_prompt_no_sample_no_instruct.csv"
)

# Tạo cột dự đoán
for col in df_2_prompts.columns:
    if col.endswith("_output"):
        pred_col = col.replace("_output", "_pred")
        df_2_prompts[pred_col] = df_2_prompts[col].apply(extract_label_from_text)

# Ground-truth labels
true_labels = df_2_prompts["label"].tolist()

results_metrics = []
for col in df_2_prompts.columns:
    if col.endswith("_pred"):
        pred_labels = df_2_prompts[col].tolist()

        # Lọc chỉ giữ các mẫu có cả true & pred thuộc nhãn hợp lệ
        filtered_data = [
            (t, p) for t, p in zip(true_labels, pred_labels)
            if (t in VALID_LABELS) and (p in VALID_LABELS)
        ]
        if not filtered_data:  # bỏ qua nếu không có mẫu hợp lệ
            continue

        y_true, y_pred = zip(*filtered_data)

        # Tính toán báo cáo classification
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=VALID_LABELS,
            digits=2,
            zero_division=0,
            output_dict=True
        )

        # Tính confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=VALID_LABELS)
        
        # Hiển thị confusion matrix
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix for {col.replace('_pred', '')}")
        plt.show()

        results_metrics.append({
            "variant": col.replace("_pred", ""),
            "accuracy": report_dict["accuracy"],
            "macro_precision": report_dict["macro avg"]["precision"],
            "macro_recall": report_dict["macro avg"]["recall"],
            "macro_f1": report_dict["macro avg"]["f1-score"],
        })

# ========== Xuất kết quả ========== 
metrics_df = pd.DataFrame(results_metrics)
metrics_df.to_csv(
    "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/cot2_detail/qwen_2.5_14B_new_prompt_no_sample_no_instruct_metrics.csv",
    index=False,
    encoding="utf-8"
)



# ###################################################################################################
# # import json
# # import re

# # def extract_label(text):
# #     match = re.search(r'\b(Non HS|Implicit HS|Explicit HS)\b', str(text))
# #     if match:
# #         return match.group(1)
# #     return None

# # def keep_if_latin(text):
# #     if text is None:
# #         return None
# #     if all('a' <= c.lower() <= 'z' or c.isspace() for c in text):
# #         return text
# #     return None

# # data = []

# # input_file = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/generated_data.jsonl"
# # with open(input_file, "r", encoding="utf-8") as f:
# #     lines = [json.loads(line) for line in f]

# # for line in lines:
# #     # Chuẩn hóa nhãn
# #     label = extract_label(line.get("label", ""))
# #     if label is None:
# #         continue  # bỏ nếu không có nhãn hợp lệ
# #     line["label"] = label

# #     # Giữ lại text nếu toàn bộ là chữ Latin (nếu cần)
# #     if "text" in line:
# #         line["text"] = keep_if_latin(line["text"]) or line["text"]

# #     data.append(line)

# # out_file = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/generated_data_official.jsonl"
# # with open(out_file, "w", encoding="utf-8") as f:
# #     for sample in data:
# #         f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# # print("✅ Hoàn tất: sinh ~", len(data), "samples và lưu vào", out_file)
