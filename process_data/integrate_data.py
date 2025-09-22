import json

new_data = []

with open("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/translated_explicit_hate_speech.jsonl",'r',encoding='utf-8') as f:
    for line in f:
        new_data.append(json.loads(line))

with open("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/translated_implicit_hate_speech.jsonl",'r',encoding='utf-8') as f:
    for line in f:
        new_data.append(json.loads(line))

with open("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/translated_non_hate_speech.jsonl",'r',encoding='utf-8') as f:
    for line in f:
        new_data.append(json.loads(line))


import random

random.shuffle(new_data)

with open("/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/data/official_test.jsonl",'w',encoding='utf-8') as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')