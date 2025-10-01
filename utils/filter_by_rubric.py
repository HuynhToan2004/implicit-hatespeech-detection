import pandas as pd

# Đọc dữ liệu
df = pd.read_csv('./data/rubric/TOXIGEN/after_voting.csv')

# Lọc các sample cho test với các điều kiện cụ thể
df_filtered_test = df[(df['Semantic_Fidelity'] == 3) & 
                       (df['Toxicity_Preservation'] == 3) & 
                       (df['Cultural_Fit'] == 3)].reset_index(drop=True)

# Chọn đủ 400 sample cho test
df_filtered_test = df_filtered_test.head(400)

# Lưu file test
df_filtered_test.to_csv("./data/test/TOXIGEN_Imp_test.csv", index=False, encoding='utf-8-sig')

# Lọc các sample cho train với điều kiện cụ thể
df_filtered_train = df[(df['Semantic_Fidelity'] >= 3) & 
                       (df['Toxicity_Preservation'] >= 2) & 
                       (df['Cultural_Fit'] >= 2)].reset_index(drop=True)

# Sử dụng index mặc định để so sánh
df_filtered_train = df_filtered_train[~df_filtered_train.index.isin(df_filtered_test.index)]

# Lưu file train
df_filtered_train.to_csv("./data/train/TOXIGEN_Imp_train.csv", index=False, encoding='utf-8-sig')

print(f"Tổng số sample trong test: {df_filtered_test.shape[0]}")
print(f"Tổng số sample trong train: {df_filtered_train.shape[0]}")
