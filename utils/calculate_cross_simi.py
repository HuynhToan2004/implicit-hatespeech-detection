import pandas as pd
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer, util


with open('./data/TOXIGEN_Imp.jsonl', 'r', encoding='utf-8') as infile:
    data_list = [json.loads(line) for line in infile]

df = pd.DataFrame(data_list)

# Tải mô hình đã lưu trên local
model = SentenceTransformer('/data2/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/paraphrase-multilingual-mpnet-base-v2')  
similarities = []

# Tính similarity cho mỗi cặp câu
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating similarity"):
    source = row['original']
    translation = row['translation']
    
    # Tính vector cho source và translation
    source_embedding = model.encode(source, convert_to_tensor=True)
    translation_embedding = model.encode(translation, convert_to_tensor=True)
    
    # Tính độ tương tự giữa hai câu (Cosine similarity)
    similarity = util.pytorch_cos_sim(source_embedding, translation_embedding)[0][0].item()
    
    similarities.append(similarity)

average_similarity = sum(similarities) / len(similarities)

# Thêm cột similarity vào DataFrame
df['similarity'] = similarities

# Lọc những câu có similarity >= 0.8
df_filtered = df[df['similarity'] >= 0.8].reset_index(drop=True)

# Lưu kết quả vào file CSV
df_filtered.to_csv("./data/TOXIGEN_Imp_after_cross_simi.csv", index=False, encoding='utf-8-sig')

# In kết quả
print(f"Tổng số câu: {len(similarities)}")
print(f"Similarity trung bình: {average_similarity:.4f}")