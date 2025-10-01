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
        if item.get('is_definition_accurate') == 'yes' and len(item.get('original', '').split()) >= 5
    ]

    with open(f'{path}_filtered.jsonl', 'w', encoding='utf-8-sig') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')