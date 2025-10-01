

import pandas as pd
from scipy import stats

# Đọc các file CSV đầu vào
file1 = pd.read_csv("/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/rubric/TOXIGEN/TOXIGEN_Imp_rubric_gemini.csv")
file2 = pd.read_csv("/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/rubric/TOXIGEN/TOXIGEN_Imp_rubric_gpt.csv")
file3 = pd.read_csv("/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/rubric/TOXIGEN/TOXIGEN_Imp_rubric_mistral.csv")

output_df = pd.DataFrame()

output_df['original'] = file1['original']
output_df['translation'] = file1['translation']
output_df['target'] = file1['target']
output_df['label'] = file1['label']
output_df['similarity'] = file1['similarity']

output_df['Semantic_Fidelity'] = stats.mode([file1['Semantic_Fidelity'], file2['Semantic_Fidelity'], file3['Semantic_Fidelity']], axis=0)[0]
output_df['Toxicity_Preservation'] = stats.mode([file1['Toxicity_Preservation'], file2['Toxicity_Preservation'], file3['Toxicity_Preservation']], axis=0)[0]
output_df['Cultural_Fit'] = stats.mode([file1['Cultural_Fit'], file2['Cultural_Fit'], file3['Cultural_Fit']], axis=0)[0]

# Lưu kết quả vào file output
output_df.to_csv('/data2/npl/luannt/IHSD/implicit-hatespeech-detection/data/rubric/TOXIGEN/after_voting.csv', index=False)

print("Major voting đã được thực hiện và kết quả đã được lưu vào './rubric/TOXIGEN/after_voting.csv'")
