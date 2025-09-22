from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import json

# ✅ Load mô hình
model = AutoModelForSequenceClassification.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/phobert-finetune-hatespeech", num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/phobert-finetune-hatespeech")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)


# ✅ Đọc file .jsonl
data_list = []
with open('/data/npl/ICEK/VACNIC/data/backup/LLMs/data.jsonl', 'r', encoding='utf-8') as infile:
    for line in infile:
        item = json.loads(line)
        data_list.append({
            'text': item['translation'],
            'label': item['label']
        })

df = pd.DataFrame(data_list)

# ✅ Ánh xạ nhãn gốc thành nhãn thực (ground truth)
df['true_label'] = df['label'].map(
    lambda x: 'LABEL_2' if x == 'implicit' else ('LABEL_2' if x == 'explicit' else 'LABEL_0')
)

# ✅ Dự đoán với pipeline
preds = []
for text in tqdm(df['text'], desc="Đang phân loại one-shot"):
    # print(pipe(text)[0][0]['label'])
    result = pipe(text)[0][0]['label']  # Lấy label đầu tiên (cao nhất)
    preds.append(result)

df['prediction'] = preds

# ✅ Đánh giá kết quả
print("📊 Breakdown dự đoán:")
print(df['prediction'].value_counts())
print("\n📊 Evaluation report:")
print(classification_report(df['true_label'], df['prediction']))