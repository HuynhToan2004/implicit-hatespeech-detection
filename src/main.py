
import argparse
import os
import sys
sys.path.append('/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection')
from pathlib import Path

from utils import load_yaml, load_llm
from templates import  (make_zero_prompt, 
                        make_fewshot_prompt_with_available_exp,
                        generate_few_shot_examples, 
                        make_prompt_for_self_generative_fewshot,
                        make_CoT_prompt,
                        make_CoT_two_prompts,
                        make_CoT_prompt_scen2,
                        make_CoT_prompt_scen3,
                        make_CoT_prompt_scen1
)
from chatbase import ChatAgent
from templates import ZEROSHOT_PROMPT, FEWSHOT_PROMPT  
from tqdm import tqdm
import json 


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ------------------------------------------------------------------ #
def parse_args():
    p = argparse.ArgumentParser("Implicit hate detector")
    p.add_argument("-c", "--config", default="config.yml")
    p.add_argument("-m", "--model", default="llama2",
                #    choices=["llama2", "qwen", "mistral", "gemma"]
                   )
    p.add_argument("-scene", "--scenario",default='cot', choices=["cot", "cot_scen1", "cot_scen2", "cot_scen3"])

    p.add_argument("-p", "--prompt-type", default="zero",
                   choices=["zero", "few", "self_few", "cot","cot2", "self_consistent"])
    p.add_argument("--sc-samples", type=int, default=11,
                   help="Số mẫu sampling để bỏ phiếu (>=3).")
    p.add_argument("--sc-temperature", type=float, default=0.7,
                   help="Nhiệt độ sampling (nên >0).")
    p.add_argument("--sc-top-p", type=float, default=0.95,
                   help="Top-p sampling.")
    p.add_argument("-i", "--input-file", required=True,
                   help="Đường dẫn file .jsonl cần infer")
    p.add_argument("-o", "--output-file", default="output.jsonl")
    # p.add_argument("--show-output", action="store_true")
    return p.parse_args()

# ------------------------------------------------------------------ #
def main():
    args = parse_args()
    import torch   
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device set to use {Device}")
    cfg = load_yaml(Path(args.config))
    if args.model not in cfg["models"]:
        sys.exit(f"✗ Model “{args.model}” not in {args.config}")
    cfg_model = cfg["models"][args.model]
    if args.prompt_type == "self_consistent":
        cfg_model["do_sample"] = True
        cfg_model["temperature"] = args.sc_temperature
        cfg_model["top_p"] = args.sc_top_p

    _, _, _, llm_pipe = load_llm(cfg_model)          
    agent = ChatAgent(llm_pipe)
    ################################## Create prompt based on prompt_type######################333
    if args.prompt_type == "zero":
        prompt_template = make_zero_prompt(args.model)
    elif args.prompt_type == "few":
        prompt_template = make_fewshot_prompt_with_available_exp(args.model)
    elif args.prompt_type == "cot":
        if args.scenario == "cot":
            prompt_template = make_CoT_prompt(args.model)
        elif args.scenario == "cot_scen1":
            prompt_template = make_CoT_prompt_scen1(args.model)
        elif args.scenario == "cot_scen2":
            prompt_template = make_CoT_prompt_scen2(args.model)
        elif args.scenario == "cot_scen3":
            prompt_template = make_CoT_prompt_scen3(args.model)

    elif args.prompt_type == "self_consistent":
         if args.scenario == "cot":
             prompt_template = make_CoT_prompt(args.model)
         elif args.scenario == "cot_scen1":
             prompt_template = make_CoT_prompt_scen1(args.model)
         elif args.scenario == "cot_scen2":
             prompt_template = make_CoT_prompt_scen2(args.model)
         elif args.scenario == "cot_scen3":
             prompt_template = make_CoT_prompt_scen3(args.model)


    with open(args.input_file, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)

    with open(args.input_file, "r", encoding="utf-8") as fin, \
         open(args.output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Infer",total=n_lines, unit="line"):
            record = json.loads(line)
            text_vi = record["translation"]
            
            if args.prompt_type == "zero":
                pred_label, full_out = agent.inference_from_raw(
                    prompt_template, {"text": text_vi}
                )
            elif args.prompt_type == "few":
                pred_label, full_out = agent.inference_few_shot(
                    examples       = prompt_template["examples"],
                    example_template = prompt_template["example_tpl"],
                    suffix         = prompt_template["suffix"],
                    input_values   = {"text": text_vi},
                )
            elif args.prompt_type == "self_few":
                generated_examples, response_text = generate_few_shot_examples(agent, args.model, n=10)
                print('generated_examples: ', generated_examples)
                prompt_template = make_prompt_for_self_generative_fewshot(args.model, generated_examples)
                if not generated_examples:
                    prompt_template = make_fewshot_prompt_with_available_exp(args.model)
                pred_label, full_out = agent.inference_few_shot(
                    examples       = prompt_template["examples"],
                    example_template = prompt_template["example_tpl"],
                    suffix         = prompt_template["suffix"],
                    input_values   = {"text": text_vi},
                )
            
            elif args.prompt_type == "cot":
                pred_label, full_out = agent.inference_from_raw(
                    prompt_template, {"text": text_vi}
                )

            elif args.prompt_type == "cot2":
                reason_tpl, label_tpl = make_CoT_two_prompts(args.model)
                pred_label, full_out, reasoning = agent.inference_two_step(
                    reason_tpl, label_tpl, {"text": text_vi}
                )
                print(f"Reasoning (Prompt 1 output):\n{reasoning}\n{'-'*50}")
                print(f"Predicted: {pred_label}, Label: {record.get('label', 'N/A')}")
                print(f"Final Output (Prompt 2 output):\n{full_out}\n{'-'*50}")
            
            elif args.prompt_type == "self_consistent":
                 pred_label, vote_counts, raw_samples = agent.inference_self_consistency(
                     prompt_template=prompt_template,
                     input_values={"text": text_vi},
                    #  n_samples=args.sc_samples,
                    #  temperature=args.sc_temperature,
                    #  top_p=args.sc_top_p
                 )
                 full_out = {
                     "samples": raw_samples,
                     "votes":   vote_counts
                 }

            print('predicted: ',pred_label, ',label: ',record.get("label", "N/A"))
            # print('prompt type: ',args.prompt_type)
            print(full_out)
            fout.write(json.dumps({
                "translation": text_vi,
                "predicted":   pred_label,
                "label":       record.get("label", "N/A"),
                "output_llm":  full_out,
                "reasoning":   reasoning if args.prompt_type == "cot2" else None,
                "full_output": full_out if args.prompt_type == "self_consistent" else None
            }, ensure_ascii=False) + "\n")

    print(f"✓ Done. Output saved to {args.output_file}")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
