
import json 
import sys
sys.stdout.reconfigure(encoding='utf-8')



# ====================CHECK LENGTH==============================================
with open('imp_hate.jsonl','r',encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        original_length = len(data['original'])
        translation_length = len(data['translation'])
        if (original_length / translation_length) >= 2.5 or (translation_length / original_length) >= 2.5:
            print("original: ", data['original'])
            print("translation: ", data['translation'])

