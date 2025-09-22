from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
import yaml
import os
import torch


def load_yaml(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# def load_llm(config,token):
#     # load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(config['model_id'],token=token)
#     tokenizer.padding_side = "left"
#     tokenizer.add_special_tokens({
#         "eos_token": "</s>",
#         "bos_token": "<s>",
#         "unk_token": "<unk>",
#         "pad_token": "<unk>",
#     })

#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(
#         config['model_id'],
#         torch_dtype=torch.float16,
#         token=token
#     )  

#     gen_config = GenerationConfig(
#         task               = config['model_config']['task'],
#         max_new_tokens     = config['model_config']['max_new_tokens'],
#         do_sample          = config['model_config']['do_sample'],
#         temperature        = config['model_config']['temperature'],
#         top_p              = config['model_config']['top_p'],
#         repetition_penalty = config['model_config']['repetition_penalty'],
#         num_beams          = config['model_config']['num_beams'],        
#         use_cache          = config['model_config']['use_cache']
#     )

#     model.generation_config = gen_config

#     text_pipeline = pipeline(
#         config['model_config']['task'],
#         model=model.cuda(config['device']),
#         tokenizer=tokenizer,
#         device=config['device']
#     )

#     llm_pipeline = HuggingFacePipeline(pipeline=text_pipeline)
#     return tokenizer, model, text_pipeline, llm_pipeline



def load_llm(config):

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_id'],
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    added = tokenizer.add_special_tokens({
        "eos_token": tokenizer.eos_token or "</s>",
        "bos_token": tokenizer.bos_token or "<s>",
        "unk_token": tokenizer.unk_token or "<unk>",
        "pad_token": tokenizer.pad_token,
    })

    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'],
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        trust_remote_code=True
    )

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # 3) generation config
    # gen_cfg_raw = config["model_config"]
    gen_config = GenerationConfig(
        max_new_tokens     = config["max_new_tokens"],
        do_sample          = config["do_sample"],
        temperature        = config["temperature"],
        top_p              = config["top_p"],
        repetition_penalty = config["repetition_penalty"],
        num_beams          = config["num_beams"],
        use_cache          = config["use_cache"],
        eos_token_id       = tokenizer.eos_token_id,
        pad_token_id       = tokenizer.pad_token_id,
    )
    model.generation_config = gen_config

    # 4) pipeline
    text_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device_map="auto",
    )
    llm_pipe = HuggingFacePipeline(pipeline=text_pipe)

    return tokenizer, model, text_pipe, llm_pipe
import re
import unicodedata
from typing import Dict, List

# ---------- 1) Chuẩn hoá nhãn đích ----------
_CANONICAL: Dict[str, str] = {
    "explicit hs":   "Explicit HS",
    "implicit hs":   "Implicit HS",
    "non hs":        "Non HS",
    "explicit hate": "Explicit HS",
    "implicit hate": "Implicit HS",
    "not hate":      "Non HS",
    "safe":          "Non HS",
    "hate":          "Explicit HS",
    "0": "Explicit HS", "1": "Implicit HS", "2": "Non HS",
    "label0": "Explicit HS", "label1": "Implicit HS", "label2": "Non HS",
}

def _norm_key(s: str) -> str:
    """Chuẩn hoá khoá tra cứu về dạng thường, bỏ gạch, gom khoảng trắng, cắt dấu câu cuối."""
    s = s.strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:! ")
    return s

# ---------- 2) Regex tìm nhãn ----------
# 2a. Câu kết luận rõ ràng (“gán nhãn là …” hoặc “label: …” ba lớp chuẩn)
_RE_CONCLUSION = re.compile(
    r"(?:gán nhãn là|kết luận\s*[:\-–]?|^\s*label\s*[:\-–]?)\s*['\"]?"
    r"(explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs)['\"]?",
    re.IGNORECASE | re.MULTILINE,
)

# 2b. Dòng có tiền tố label/prediction/… và *nhãn hợp lệ* (không cho 'hate'/'safe' trôi nổi)
_RE_LABEL_LINE = re.compile(
    r"^\s*(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*"
    r"(?P<label>explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs|"
    r"explicit[\s_\-]?hate|implicit[\s_\-]?hate|not[\s_\-]?hate|"
    r"label[012]|[012])\b",
    re.IGNORECASE | re.MULTILINE,
)

# 2c. Bắt tự do trong phần trả lời cuối, nhưng CHỈ những nhãn hợp lệ (loại 'hate'/'safe' đơn lẻ)
_RE_LABEL_ANY = re.compile(
    r"(?P<label>"
    r"explicit[\s_\-]?hs\b|implicit[\s_\-]?hs\b|non[\s_\-]?hs\b|"
    r"explicit[\s_\-]?hate\b|implicit[\s_\-]?hate\b|not[\s_\-]?hate\b|"
    r"label[012]\b|[012]\b"
    r")",
    re.IGNORECASE,
)

# 2d. Cho phép 'hate'/'safe' *chỉ khi có tiền tố* (label/prediction/…)
_RE_PREFIXED_HS_SAFE = re.compile(
    r"\b(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*(?P<label>hate\b|safe\b)",
    re.IGNORECASE,
)

# ---------- 3) Các mốc assistant ----------
_ASSIST_SEP = [
    r"\[\/INST\]",                          # Llama-2 / Mistral
    r"<\|start_header_id\|>assistant",      # Llama-3
    r"<\|im_start\|>assistant",             # Qwen
    r"\bassistant:",                        # Gemma
    r"\"role\"\s*:\s*\"assistant\"",        # OpenAI (JSON)
    r"<\|assistant\|>",                     # một số định dạng khác
]
_RE_SEP = re.compile("|".join(_ASSIST_SEP), re.IGNORECASE)

# ---------- 4) Hàm trích nhãn ----------
def extract_label(text_output: str) -> str:
    """
    Trả về: 'Explicit HS' | 'Implicit HS' | 'Non HS' | 'Unknown'
    Chiến lược:
      - Cắt lấy phần trả lời cuối của assistant.
      - Ưu tiên câu kết luận ('gán nhãn là ...' / 'label: ...' ba lớp chuẩn).
      - Sau đó ưu tiên dòng có tiền tố (label/prediction/...).
      - Nếu không có, bắt nhãn hợp lệ đầu tiên trong phần trả lời.
      - Cuối cùng mới cho phép 'hate'/'safe' nếu có tiền tố.
    """

    # 4.1 Cắt phần trả lời cuối của assistant (loại prompt & ví dụ cũ)
    for sep in (
        "<|start_header_id|>assistant",
        "<|im_start|>assistant",
        "[/INST]",
        "<|assistant|>",
    ):
        if sep in text_output:
            text_output = text_output.split(sep)[-1]

    # 4.2 Chuẩn hoá unicode; GIỮ nguyên xuống dòng để regex ^...$ hoạt động
    txt = unicodedata.normalize("NFKC", text_output)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()

    # 4.3 Nếu còn nhiều khối assistant, chỉ giữ khối cuối
    last_idx = None
    for m in _RE_SEP.finditer(txt):
        last_idx = m.end()
    if last_idx is not None:
        txt = txt[last_idx:].lstrip()

    # 4.4 Ưu tiên bắt câu kết luận rõ ràng
    m_con = _RE_CONCLUSION.search(txt)
    if m_con:
        raw = _norm_key(m_con.group(1))
        return _CANONICAL.get(raw, "Unknown")

    # 4.5 Ưu tiên dòng có tiền tố label/prediction/answer/class/nhãn
    m_line = _RE_LABEL_LINE.search(txt)
    if m_line:
        raw = _norm_key(m_line.group("label"))
        return _CANONICAL.get(raw, "Unknown")

    # 4.6 Nếu không có, lấy NHÃN HỢP LỆ ĐẦU TIÊN (tránh 'hate' rơi trong giải thích)
    m_any = _RE_LABEL_ANY.search(txt)
    if m_any:
        raw = _norm_key(m_any.group("label"))
        return _CANONICAL.get(raw, "Unknown")

    # 4.7 Cuối cùng mới cho phép 'hate'/'safe' nếu có tiền tố
    m_loose = _RE_PREFIXED_HS_SAFE.search(txt)
    if m_loose:
        raw = _norm_key(m_loose.group("label"))
        return _CANONICAL.get(raw, "Unknown")

    return "Unknown"
