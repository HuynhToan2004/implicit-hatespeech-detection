import json
from sklearn.metrics import classification_report


# for i in ['cot','cot_scen1','cot_scen2','cot_scen3']:
# for i in ['zs','fs','sfs','cot','cot2','sc']:
#     # Đường dẫn đến file .jsonl
#     try:
#         file_path = f"/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/{i}/gemma_3_12B.jsonl"
#         name = file_path.split('/')[-2:]
#         name = ['/'.join(name)][0]
#         # Danh sách để chứa nhãn thật và nhãn dự đoán
#         true_labels = []
#         predicted_labels = []

#         # Đọc dữ liệu từ file .jsonl
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 data = json.loads(line)
#                 true_labels.append(data['label'].lower())
#                 predicted_labels.append(data['predicted'].lower())

#         # Tính classification report
#         report = classification_report(true_labels, predicted_labels, zero_division=0)

#         # In kết quả
#         print(f'{name}\n', report)
#     except:
#         print(f"File {file_path} chưa có.")
#         continue

# Đường dẫn đến file .jsonl
file_path = f"/data2/npl/luannt/IHSD/implicit-hatespeech-detection/output/fs/qwen_2.5_7B.jsonl"
name = file_path.split('/')[-2:]
name = ['/'.join(name)][0]
# Danh sách để chứa nhãn thật và nhãn dự đoán
true_labels = []
predicted_labels = []

# Đọc dữ liệu từ file .jsonl
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        true_labels.append(data['label'].lower())
        predicted_labels.append(data['predicted'].lower())

# Tính classification report
report = classification_report(true_labels, predicted_labels, zero_division=0)

# In kết quả
print(f'{name}\n', report)
