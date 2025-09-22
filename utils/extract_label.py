import json
import re
import unicodedata
from typing import List

# Định nghĩa từ điển chuẩn hóa nhãn
import re
import unicodedata
from typing import Dict, List
from sklearn.metrics import classification_report


######################################################CODE extract chung#################################
# ---------- 1) Chuẩn hoá nhãn đích ----------
# _CANONICAL: Dict[str, str] = {
#     "explicit hs":   "Explicit HS",
#     "implicit hs":   "Implicit HS",
#     "non hs":        "Non HS",
#     "explicit hate": "Explicit HS",
#     "implicit hate": "Implicit HS",
#     "not hate":      "Non HS",
#     "safe":          "Non HS",
#     "hate":          "Explicit HS",
#     "0": "Explicit HS", "1": "Implicit HS", "2": "Non HS",
#     "label0": "Explicit HS", "label1": "Implicit HS", "label2": "Non HS",
# }

# def _norm_key(s: str) -> str:
#     """Chuẩn hoá khoá tra cứu về dạng thường, bỏ gạch, gom khoảng trắng, cắt dấu câu cuối."""
#     s = s.strip().lower().replace("-", " ").replace("_", " ")
#     s = re.sub(r"\s+", " ", s)
#     s = s.rstrip(".,;:! ")
#     return s

# # ---------- 2) Regex tìm nhãn ----------
# # 2a. Câu kết luận rõ ràng (“gán nhãn là …” hoặc “label: …” ba lớp chuẩn)
# _RE_CONCLUSION = re.compile(
#     r"(?:gán nhãn là|kết luận\s*[:\-–]?|^\s*label\s*[:\-–]?)\s*['\"]?"
#     r"(explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs)['\"]?",
#     re.IGNORECASE | re.MULTILINE,
# )

# # 2b. Dòng có tiền tố label/prediction/… và *nhãn hợp lệ* (không cho 'hate'/'safe' trôi nổi)
# _RE_LABEL_LINE = re.compile(
#     r"^\s*(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*"
#     r"(?P<label>explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs|"
#     r"explicit[\s_\-]?hate|implicit[\s_\-]?hate|not[\s_\-]?hate|"
#     r"label[012]|[012])\b",
#     re.IGNORECASE | re.MULTILINE,
# )

# # 2c. Bắt tự do trong phần trả lời cuối, nhưng CHỈ những nhãn hợp lệ (loại 'hate'/'safe' đơn lẻ)
# _RE_LABEL_ANY = re.compile(
#     r"(?P<label>"
#     r"explicit[\s_\-]?hs\b|implicit[\s_\-]?hs\b|non[\s_\-]?hs\b|"
#     r"explicit[\s_\-]?hate\b|implicit[\s_\-]?hate\b|not[\s_\-]?hate\b|"
#     r"label[012]\b|[012]\b"
#     r")",
#     re.IGNORECASE,
# )

# # 2d. Cho phép 'hate'/'safe' *chỉ khi có tiền tố* (label/prediction/…)
# _RE_PREFIXED_HS_SAFE = re.compile(
#     r"\b(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*(?P<label>hate\b|safe\b)",
#     re.IGNORECASE,
# )

# # ---------- 3) Các mốc assistant ----------
# _ASSIST_SEP = [
#     r"\[\/INST\]",                          # Llama-2 / Mistral
#     r"<\|start_header_id\|>assistant",      # Llama-3
#     r"<\|im_start\|>assistant",             # Qwen
#     r"\bassistant:",                        # Gemma
#     r"\"role\"\s*:\s*\"assistant\"",        # OpenAI (JSON)
#     r"<\|assistant\|>",                     # một số định dạng khác
# ]
# _RE_SEP = re.compile("|".join(_ASSIST_SEP), re.IGNORECASE)

# # ---------- 4) Hàm trích nhãn ----------
# def extract_label(text_output: str) -> str:
#     """
#     Trả về: 'Explicit HS' | 'Implicit HS' | 'Non HS' | 'Unknown'
#     Chiến lược:
#       - Cắt lấy phần trả lời cuối của assistant.
#       - Ưu tiên câu kết luận ('gán nhãn là ...' / 'label: ...' ba lớp chuẩn).
#       - Sau đó ưu tiên dòng có tiền tố (label/prediction/...).
#       - Nếu không có, bắt nhãn hợp lệ đầu tiên trong phần trả lời.
#       - Cuối cùng mới cho phép 'hate'/'safe' nếu có tiền tố.
#     """

#     # 4.1 Cắt phần trả lời cuối của assistant (loại prompt & ví dụ cũ)
#     for sep in (
#         "<|start_header_id|>assistant",
#         "<|im_start|>assistant",
#         "[/INST]",
#         "<|assistant|>",
#     ):
#         if sep in text_output:
#             text_output = text_output.split(sep)[-1]

#     # 4.2 Chuẩn hoá unicode; GIỮ nguyên xuống dòng để regex ^...$ hoạt động
#     txt = unicodedata.normalize("NFKC", text_output)
#     txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()

#     # 4.3 Nếu còn nhiều khối assistant, chỉ giữ khối cuối
#     last_idx = None
#     for m in _RE_SEP.finditer(txt):
#         last_idx = m.end()
#     if last_idx is not None:
#         txt = txt[last_idx:].lstrip()

#     # 4.4 Ưu tiên bắt câu kết luận rõ ràng
#     m_con = _RE_CONCLUSION.search(txt)
#     if m_con:
#         raw = _norm_key(m_con.group(1))
#         return _CANONICAL.get(raw, "Unknown")

#     # 4.5 Ưu tiên dòng có tiền tố label/prediction/answer/class/nhãn
#     m_line = _RE_LABEL_LINE.search(txt)
#     if m_line:
#         raw = _norm_key(m_line.group("label"))
#         return _CANONICAL.get(raw, "Unknown")

#     # 4.6 Nếu không có, lấy NHÃN HỢP LỆ ĐẦU TIÊN (tránh 'hate' rơi trong giải thích)
#     m_any = _RE_LABEL_ANY.search(txt)
#     if m_any:
#         raw = _norm_key(m_any.group("label"))
#         return _CANONICAL.get(raw, "Unknown")

#     # 4.7 Cuối cùng mới cho phép 'hate'/'safe' nếu có tiền tố
#     m_loose = _RE_PREFIXED_HS_SAFE.search(txt)
#     if m_loose:
#         raw = _norm_key(m_loose.group("label"))
#         return _CANONICAL.get(raw, "Unknown")

#     return "Unknown"


##############################################CODE EXTRACT MISTRAL##########################################3
import re
import unicodedata
from typing import Dict, Optional

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
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip("()[]{}\"'`“”‘’«»")  # bỏ ngoặc
    s = s.rstrip(".,;:! ")
    return s

# ---------- 2) Mốc & regex ----------
_RE_LAST_INST = re.compile(r"\[\/INST\]", re.IGNORECASE)

# Bắt 'label|prediction|answer|class|nhãn: <nhãn>' ở bất kỳ vị trí nào
_RE_WITH_PREFIX = re.compile(
    r"(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*"
    r"(?P<label>"
    r"explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs|"
    r"explicit[\s_\-]?hate|implicit[\s_\-]?hate|not[\s_\-]?hate|"
    r"label[012]|[012]|hate|safe"
    r")\b",
    re.IGNORECASE,
)

# Câu kết luận tiếng Việt/Anh thường gặp (không yêu cầu ở đầu dòng)
_RE_CONCLUSION_PHRASE = re.compile(
    r"(?:gán nhãn là|kết luận|phân loại|classification|label)\s*[:\-–]?\s*['\"]?"
    r"(?P<label>explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs|label[012]|[012])\b",
    re.IGNORECASE,
)

# Quét tự do *chỉ* 3 lớp HS & số (không bắt hate/safe trôi nổi)
_RE_FREE_STRICT = re.compile(
    r"(?P<label>explicit[\s_\-]?hs|implicit[\s_\-]?hs|non[\s_\-]?hs|label[012]|[012])\b",
    re.IGNORECASE,
)

# Nếu bất đắc dĩ mới cho 'hate/safe' nhưng phải có tiền tố
_RE_HS_SAFE_PREFIXED = re.compile(
    r"(?:label|prediction|answer|class|nhãn)\s*[:\-–]?\s*(?P<label>hate|safe)\b",
    re.IGNORECASE,
)

# Một số dấu mốc assistant khác (phòng hờ)
_RE_SEP = re.compile(
    r"(?:<\|start_header_id\|>assistant|<\|im_start\|>assistant|<\|assistant\|>|\"role\"\s*:\s*\"assistant\")",
    re.IGNORECASE,
)

def _cut_after_last_inst(txt: str) -> str:
    """Cắt mọi thứ trước dấu [/INST] cuối cùng. Nếu không có, thử mốc assistant khác."""
    m = None
    for mm in _RE_LAST_INST.finditer(txt):
        m = mm
    if m:
        return txt[m.end():]

    # Thử các mốc assistant khác (ít khi cần)
    last = None
    for mm in _RE_SEP.finditer(txt):
        last = mm
    if last:
        return txt[last.end():]
    return txt

def _pick_last(pattern: re.Pattern, txt: str) -> Optional[str]:
    last = None
    for m in pattern.finditer(txt):
        last = m
    if last:
        return last.group("label")
    return None

# ---------- 3) Hàm trích nhãn ----------
def extract_label(text_output: str) -> str:
    """
    Trả về: 'Explicit HS' | 'Implicit HS' | 'Non HS' | 'Unknown'
    Ưu tiên (đều tính trên phần *sau* [/INST] cuối):
      1) Dòng có tiền tố (label/prediction/answer/class/nhãn)
      2) Câu kết luận (gán nhãn/label/classification ...)
      3) Quét tự do STRICT (chỉ 3 lớp & số)
      4) Cuối cùng mới cho 'hate/safe' nhưng bắt buộc có tiền tố
      -> Trong mỗi bước, lấy *match CUỐI CÙNG*.
    """
    # 1) Chuẩn hoá & cắt
    txt = unicodedata.normalize("NFKC", text_output)
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = _cut_after_last_inst(txt).strip()

    # 2) Ưu tiên theo thứ tự, luôn chọn match CUỐI CÙNG
    for pat in (_RE_WITH_PREFIX, _RE_CONCLUSION_PHRASE, _RE_FREE_STRICT, _RE_HS_SAFE_PREFIXED):
        raw = _pick_last(pat, txt)
        if raw:
            key = _norm_key(raw)
            return _CANONICAL.get(key, "Unknown")

    return "Unknown"


###############################################################

# Hàm xử lý file JSONL
def update_jsonl_labels(input_file: str, output_file: str) -> None:
    """
    Đọc file JSONL, trích xuất nhãn từ output_llm, gán vào predicted, và lưu vào file mới.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Đọc dòng JSON
            data = json.loads(line.strip())
            
            # Trích xuất nhãn từ output_llm
            extracted_label = extract_label(data.get("output_llm", ""))
            
            # Cập nhật trường predicted
            data["predicted"] = extracted_label
            
            # Ghi lại dòng JSON đã cập nhật
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

def classification_report1(file_path):
    name = file_path.split('/')[-2:]
    name = ['/'.join(name)][0]
    # Danh sách để chứa nhãn thật và nhãn dự đoán
    true_labels = []
    predicted_labels = []

    # Đọc dữ liệu từ file .jsonl
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            true_labels.append(data['label'].lower())
            predicted_labels.append(data['predicted'].lower())

    # Tính classification report
    report = classification_report(true_labels, predicted_labels, zero_division=0)

    # In kết quả
    print(f'{name}\n', report)

# Sử dụng hàm
if __name__ == "__main__":
    input_file = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/zs/mistral_7B_v0.2.jsonl" 
    output_file = "/data2/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/output/zs/mistral_7B_v0.2_re.jsonl"  # File đầu ra
    update_jsonl_labels(input_file, output_file)
    # print(f"Đã cập nhật nhãn và lưu vào {output_file}")

    classification_report1(input_file)
    classification_report1(output_file)