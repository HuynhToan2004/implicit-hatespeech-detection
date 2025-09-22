from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
from sklearn.metrics import classification_report
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/ViHateT5-base-HSD")
model = AutoModelForSeq2SeqLM.from_pretrained("/data/npl/ICEK/VACNIC/src/data/assest/ViHateT5-base-HSD")
model.eval()

import re
EXPLICIT_SLURS = {
    # ── Từ đơn ──
    "đm", "dm", "đmm", "đcm", "đcl",
    "địt", "đụ", "đĩ", "điếm", "lồn", "cặc",
    "chó", "khỉ", "lợn", "mọi",

    # ── Cụm ≥2 từ (giữ dấu cách) ──
    "địt mẹ", "đụ mẹ", "đụ má", "địt má", "đụ mẹ mày", "địt mẹ mày",
    "con chó", "lũ chó", "bọn chó", "chó chết", "chó má",
    "con khỉ", "lũ khỉ", "bọn khỉ",
    "con lợn", "lũ lợn", "bọn lợn",
    "con mọi", "thằng mọi", "bọn mọi", "lũ mọi", "mọi đen", "đám mọi", "đồ mọi", "mọi rợ",
    "con đĩ", "con điếm", "đồ đĩ", "đồ điếm", "đĩ rẻ", "đĩ chúa"
}

data_list = []
with open('/data/npl/ICEK/VACNIC/data/backup/LLMs/official_test.jsonl', 'r', encoding='utf-8') as infile:
    i = 0
    for line in infile:
        item = json.loads(line)
        # Chỉ lấy translation và label
        data_list.append({
            'text': item['translation'],
            'label': item['label']
        })
    
df_data = pd.DataFrame(data_list)
def is_explicit(text: str) -> bool:
    """
    Trả về True nếu text chứa bất kỳ slur nằm trong EXPLICIT_SLURS.
    Không phân biệt hoa thường; tách từ bằng regex \w+.
    """
    tokens = re.findall(r"\w+", text.lower())
    return any(tok in EXPLICIT_SLURS for tok in tokens)
 
def generate_output(input_text, prefix='hate-speech-detection'):
    prefixed_input_text = prefix + ': ' + input_text
    input_ids = tokenizer.encode(prefixed_input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=256)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text
 
def zero_shot_prediction_df(df, prefix='hate-speech-detection'):
    preds = []
    for text in tqdm(df['text'], desc="Đang sinh dự đoán"):
        output = generate_output(text, prefix)
        preds.append(output.lower().strip())

    df = df.copy()                           
    df['prediction'] = preds
    df['true_label'] = df['label'].map(
        lambda x: 'hate' if x in ('Implicit HS', 'Explicit HS') else 'clean'
    )

    def get_final_pred(row):
        label = row['label']        # 'Implicit HS' / 'Explicit HS' / 'non'
        pred  = row['prediction']   # 'hate' / 'clean'
        text  = row['text']

        if label == 'Implicit HS' and pred == 'hate':
            return 'Implicit HS'
        elif label == 'Explicit HS' and pred == 'hate':
            return 'Explicit HS'

        # hate thật nhưng mô hình bảo 'clean'
        elif label in ('Implicit HS', 'Explicit HS') and pred == 'clean':
            return 'Non HS'

        # non thật + mô hình clean
        elif label == 'Non HS' and pred == 'clean':
            return 'Non HS'

        # ---- TH còn lại: label = non & pred = hate ----
        else: return 'Explicit HS' if is_explicit(text) else 'Implicit HS'

    df['final_prediction'] = df.apply(get_final_pred, axis=1)
    return df

df_zero = zero_shot_prediction_df(df_data)

print(classification_report(df_zero['label'],df_zero['final_prediction']))