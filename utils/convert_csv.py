import json 
import pandas
with open("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/cot_scen2/300_qwen_2.5_14B.jsonl",'r',encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

df = pandas.DataFrame(data)
df.to_csv("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/cot_scen2/300_qwen_2.5_14B.csv", index=False, encoding='utf-8')