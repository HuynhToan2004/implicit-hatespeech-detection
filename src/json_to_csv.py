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

# import json

# def convert_jsonl_to_json(jsonl_path, json_path):
#     data = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():  # Bỏ qua dòng trống
#                 data.append(json.loads(line))

#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

#     print(f"Đã chuyển đổi {len(data)} dòng từ {jsonl_path} -> {json_path}")

# # Ví dụ sử dụng:
# convert_jsonl_to_json("/data/npl/ICEK/DATASET/content/vacnic/train_vacnic_final.jsonl", "/data/npl/ICEK/DATASET/content/vacnic/train_vacnic_final.json")

########################################################################################

# import json
# import csv

# def jsonl_to_csv(jsonl_file, csv_file):
#     with open(jsonl_file, 'r', encoding='utf-8') as jsonl_f:
#         with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
#             writer = csv.writer(csv_f)
#             first_line = True
#             for line in jsonl_f:
#                 data = json.loads(line)
#                 if first_line:
#                     # Ghi tiêu đề từ khóa trong JSON
#                     writer.writerow(data.keys())
#                     first_line = False
#                 # Ghi các giá trị của JSON vào CSV
#                 writer.writerow(data.values())

# # Sử dụng hàm
# jsonl_to_csv('input.jsonl', 'output.csv')


#######################################################################################


import csv
import json


def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r', encoding='utf-8-sig') as csv_f:  # Đảm bảo mở với utf-8-sig
        reader = csv.DictReader(csv_f)
        with open(jsonl_file, 'w', encoding='utf-8') as jsonl_f:
            for row in reader:
                # Sử dụng ensure_ascii=False để không mã hóa lại các ký tự Unicode
                jsonl_f.write(json.dumps(row, ensure_ascii=False) + '\n')

# Sử dụng hàm
csv_to_jsonl('/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/imp/ISHate_Imp_test.csv',
              '/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/imp/ISHate_Imp_test.jsonl')




################################################################################################


import json

paths = [
    '/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/non/non_hs_gemini.jsonl',
    '/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/non/non_hs_mistral.jsonl',
    '/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/test/non/non_hs_gpt4o.jsonl'
]

for path in paths:
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = [json.loads(line) for line in f if line.strip()]

    filtered_data = [
        item for item in data
        if item.get('is_definition_accurate') == 'yes' and len(item.get('original', '')) >= 5
    ]

    with open(f'{path}_filtered.jsonl', 'w', encoding='utf-8-sig') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')