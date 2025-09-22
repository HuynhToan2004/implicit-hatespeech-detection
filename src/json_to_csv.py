# import pandas as pd
# import json
# from pathlib import Path

# def jsonl_to_csv(jsonl_path: str, csv_path: str, keep_cols=None):
#     """
#     Chuyển file .jsonl sang .csv.
    
#     Parameters
#     ----------
#     jsonl_path : str
#         Đường dẫn tới file .jsonl.
#     csv_path : str
#         Đường dẫn muốn lưu file .csv.
#     keep_cols : list[str] | None
#         Danh sách cột cần giữ lại (theo key trong JSON). 
#         Nếu None → giữ tất cả.
#     """
#     records = []
#     with Path(jsonl_path).open(encoding="utf-8") as f:
#         for line in f:
#             if not line.strip():
#                 continue  # bỏ qua dòng trống
#             obj = json.loads(line)
#             if keep_cols:
#                 obj = {k: obj.get(k) for k in keep_cols}
#             records.append(obj)

#     df = pd.DataFrame(records)
#     df.to_csv(csv_path, index=False, encoding="utf-8")
#     print(f"✓ Saved {len(df)} rows to {csv_path}")

# # --- cách dùng -----------------------------------------------------------
# # Ví dụ chỉ giữ ba cột quan trọng:
# jsonl_to_csv(
#     jsonl_path="/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/fs_qwen_2.5_14B.jsonl",
#     csv_path="/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/fs_qwen_2.5_14B.csv",
#     keep_cols=["translation", "label", "predicted", "output_llm"]
# )

import json

def convert_jsonl_to_json(jsonl_path, json_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Bỏ qua dòng trống
                data.append(json.loads(line))

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Đã chuyển đổi {len(data)} dòng từ {jsonl_path} -> {json_path}")

# Ví dụ sử dụng:
convert_jsonl_to_json("/data/npl/ICEK/DATASET/content/vacnic/train_vacnic_final.jsonl", "/data/npl/ICEK/DATASET/content/vacnic/train_vacnic_final.json")
