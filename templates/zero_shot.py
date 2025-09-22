

ZEROSHOT_PROMPT = """[INST]
    Bạn là một chuyên gia ngôn ngữ có khả năng phân tích các phát ngôn. Hãy phân loại văn bản sau thành một trong ba nhãn sau, **không giải thích**:

    Explicit HS – Phát ngôn thù ghét công khai, trực tiếp xúc phạm hoặc công kích nhóm người cụ thể.
    Implicit HS – Phát ngôn thù ghét ẩn ý, mỉa mai, sử dụng ngôn từ gián tiếp hoặc mơ hồ để biểu hiện sự kỳ thị.
    Non HS – Không phải phát ngôn thù ghét, mang tính trung lập hoặc tích cực.
    [/INST]
    [INPUT]
    Văn bản: {sentence}
    [OUTPUT]
    label:
    """