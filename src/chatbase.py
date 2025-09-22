from icecream import ic
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from utils import extract_label

class ChatAgent:
    def __init__(self, model):
        self.llm_model = model

    def _postprocess(self, chain_result):
        if isinstance(chain_result, dict):
            return chain_result.get("text", "")
        return str(chain_result)

    def inference(self, prompt_template, input_values):
        if isinstance(prompt_template, str):

            prompt_text = prompt_template.format(**input_values)
            print(f"[Prompt length: {len(prompt_text.split())} tokens]")
            return self._postprocess(self.llm_model.invoke(prompt_text))

        chain = prompt_template | self.llm_model
        print(f"[Prompt length: {len(prompt_template.format(**input_values).split())} tokens]")
        result = chain.invoke(input_values)
        return self._postprocess(result)

    def inference_from_raw(self, template_str, input_values):
        prompt_template = PromptTemplate(
            input_variables=list(input_values.keys()), template=template_str
        )
        output = self.inference(prompt_template, input_values)
        return extract_label(output), output       # <— (label, raw)

    def inference_few_shot(self, examples, example_template, suffix, input_values):
    
        if not examples:
            prompt_template = PromptTemplate(
                input_variables=list(input_values.keys()),
                template=suffix
            )
            output = self.inference(prompt_template, input_values)
            return extract_label(output), output

        example_prompt = PromptTemplate(
            input_variables=list(examples[0].keys()),
            template=example_template
        )
        prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix=suffix,
            input_variables=list(input_values.keys()),
        )
        output = self.inference(prompt_template, input_values)
        return extract_label(output), output

    def inference_self_consistency(
        self,
        prompt_template,
        input_values,
        n_samples: int = 7,
    ):
        """
        Self-consistency: model đã được cấu hình sampling từ trước
        nên KHÔNG bind() nữa.
        Trả về (best_label, vote_counts, raw_outputs).
        """
        assert n_samples >= 3, "n_samples nên >=3 để self-consistency hữu dụng."

        # 1) Render prompt
        prompt_text = (
            prompt_template.format(**input_values)
            if not isinstance(prompt_template, str)
            else prompt_template.format(**input_values)
        )

        # 2) Dùng luôn model đã mang temperature/top_p/do_sample
        llm = self.llm_model
        prompts = [prompt_text] * n_samples
        raw_outputs: list[str] = []

        # 3) Ưu tiên batch()
        try:
            if hasattr(llm, "batch"):
                outs = llm.batch(prompts)
                raw_outputs = [self._postprocess(o) for o in outs]
            else:
                raise AttributeError
        except Exception:
            # fallback tuần tự
            for _ in range(n_samples):
                raw_outputs.append(self._postprocess(llm.invoke(prompt_text)))

        # 4) Đếm phiếu
        from collections import Counter
        labels = [extract_label(o) or "UNKNOWN" for o in raw_outputs]
        vote_counts = dict(Counter(labels))

        # 5) Chọn nhãn
        max_vote = max(vote_counts.values())
        cands = [lab for lab, v in vote_counts.items() if v == max_vote]
        cands = [c for c in cands if c != "UNKNOWN"] or cands
        best_label = sorted(cands)[0]

        print(f"[Self-Consistency] votes={vote_counts} → chosen={best_label}")
        return best_label, vote_counts, raw_outputs
    
    import re

    # def inference_two_step(self, context_template, label_template, input_values):
    #     # Bước 1: Tạo phân tích từng bước
    #     context_prompt = PromptTemplate(
    #         input_variables=list(input_values.keys()),
    #         template=context_template
    #     )
    #     context_output = self.inference(context_prompt, input_values)
        
    #     # In context_output để debug
    #     print(f"Debug context_output:\n{context_output}\n{'-'*50}")
        
    #     # Loại bỏ phần hệ thống và user prompt khỏi context_output
    #     cleaned_output = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\s*<\|im_start\|>user.*?<\|im_end\|>\s*<\|im_start\|>assistant\s*", "", context_output, flags=re.DOTALL)
    #     cleaned_output = cleaned_output.replace("<|im_end|>", "").strip()
        
    #     # Trích xuất các bước bằng regex
    #     steps = {}
    #     patterns = {
    #         "Bước 1": r"- Bước 1:.*?(?=- Bước 2:|$)",
    #         "Bước 2": r"- Bước 2:.*?(?=- Bước 3:|$)",
    #         "Bước 3": r"- Bước 3:.*?(?=- Bước 4:|$)",
    #         "Bước 4": r"- Bước 4:.*?$"
    #     }
        
    #     for step, pattern in patterns.items():
    #         match = re.search(pattern, cleaned_output, re.DOTALL)
    #         if match:
    #             steps[step] = match.group(0).replace("- " + step + ":", "").strip()
    #         else:
    #             steps[step] = f"Không tìm thấy nội dung cho {step.lower()}."

    #     # Kiểm tra xem có trích xuất được ít nhất một bước không
    #     # if all("Không tìm thấy" in steps[step] for step in steps):
    #     #     print(f"Warning: Không thể trích xuất bất kỳ bước nào từ context_output:\n{context_output}")
    #     #     steps = {
    #     #         "Bước 1": "Không có từ ngữ xúc phạm.",
    #     #         "Bước 2": "Không có xúc phạm trực tiếp hay gián tiếp.",
    #     #         "Bước 3": "Sắc thái ngữ nghĩa không rõ do thiếu dữ liệu.",
    #     #         "Bước 4": "Mục đích không rõ do thiếu dữ liệu."
    #     #     }

    #     # Bước 2: Chèn phân tích vào label_template và phân loại nhãn
    #     label_input_values = input_values.copy()
    #     label_input_values.update({
    #         "Bước 1": steps["Bước 1"],
    #         "Bước 2": steps["Bước 2"],
    #         "Bước 3": steps["Bước 3"],
    #         "Bước 4": steps["Bước 4"]
    #     })
    #     print(f"Debug label_input_values:\n{label_input_values}\n{'-'*50}")
        
    #     label_prompt = PromptTemplate(
    #         input_variables=list(label_input_values.keys()),
    #         template=label_template
    #     )
    #     final_output = self.inference(label_prompt, label_input_values)
        
    #     # Trích xuất nhãn từ final_output
    #     label = extract_label(final_output)
    #     return label, final_output, context_output
    import re

    def inference_two_step(self, context_template, label_template, input_values):
        # Bước 1: Tạo phân tích từng bước
        context_prompt = PromptTemplate(
            input_variables=list(input_values.keys()),
            template=context_template
        )
        context_output = self.inference(context_prompt, input_values)
        
        # In context_output để debug
        print(f"Debug context_output:\n{context_output}\n{'-'*50}")
        
        # Loại bỏ phần hệ thống và user prompt khỏi context_output
        # Hỗ trợ cả định dạng <|im_start|> (Qwen) và <s>[INST] (Mistral)
        cleaned_output = re.sub(
            r"(<\|im_start\|>system.*?<\|im_end\|>\s*<\|im_start\|>user.*?<\|im_end\|>\s*<\|im_start\|>assistant\s*|"
            r"<s>\[INST\].*?\[/INST\]\s*)",
            "",
            context_output,
            flags=re.DOTALL
        )
        cleaned_output = cleaned_output.replace("<|im_end|>", "").replace("[/INST]", "").strip()
        
        # Trích xuất các bước bằng regex linh hoạt hơn
        steps = {}
        patterns = {
            "Bước 1": r"- Bước 1:.*?(?=(?:- Bước 2:|$))",
            "Bước 2": r"- Bước 2:.*?(?=(?:- Bước 3:|$))",
            "Bước 3": r"- Bước 3:.*?(?=(?:- Bước 4:|$))",
            "Bước 4": r"- Bước 4:.*?$"
        }
        
        for step, pattern in patterns.items():
            match = re.search(pattern, cleaned_output, re.DOTALL)
            if match:
                step_content = match.group(0).replace("- " + step + ":", "").strip()
                steps[step] = step_content if step_content else f"Không tìm thấy nội dung rõ ràng cho {step.lower()}."
            else:
                steps[step] = f"Không tìm thấy nội dung cho {step.lower()}."
                print(f"Warning: Không thể trích xuất {step} từ context_output:\n{cleaned_output}")
        
        # Kiểm tra và xử lý trường hợp không trích xuất được bước nào
        # if all("Không tìm thấy nội dung" in steps[step] for step in steps):
        #     print(f"Error: Không thể trích xuất bất kỳ bước nào từ context_output:\n{context_output}")
        #     steps = {
        #         "Bước 1": "Không có từ ngữ xúc phạm.",
        #         "Bước 2": "Không có xúc phạm trực tiếp hay gián tiếp.",
        #         "Bước 3": "Sắc thái ngữ nghĩa không rõ do thiếu dữ liệu.",
        #         "Bước 4": "Mục đích không rõ do thiếu dữ liệu."
        #     }
        
        # Thêm log chi tiết để kiểm tra các bước trích xuất
        print(f"Debug extracted steps:\n{steps}\n{'-'*50}")
        
        # Bước 2: Chèn phân tích vào label_template và phân loại nhãn
        label_input_values = input_values.copy()
        label_input_values.update({
            "Bước 1": steps["Bước 1"],
            "Bước 2": steps["Bước 2"],
            "Bước 3": steps["Bước 3"],
            "Bước 4": steps["Bước 4"]
        })
        print(f"Debug label_input_values:\n{label_input_values}\n{'-'*50}")
        
        label_prompt = PromptTemplate(
            input_variables=list(label_input_values.keys()),
            template=label_template
        )
        final_output = self.inference(label_prompt, label_input_values)
        
        # Trích xuất nhãn từ final_output
        label = extract_label(final_output)
        return label, final_output, context_output

class Prompt():
    def __init__(self, template:str,  input_values: dict):
        self.template = template
        self.input_values = input_values
        self.prompt_template = template


    def create_prompt_template(self):
        self.prompt_template = PromptTemplate(
            input_variables=self.input_values.keys(),
            template=self.template,
        )
    
    def get_prompt(self):
        return self.prompt_template.format(**self.input_values)
    
    def create_fewshot_template(self, examples: list, suffix="", prefix=""):
        self.create_prompt_template()
        self.prompt_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.prompt_template,
            prefix=prefix,
            suffix=suffix,
            input_variables=[],
        )
