"""Ánh xạ ImageNet-1K sang 16 cue categories của Geirhos et al. (2019).

Danh sách chỉ số được sao chép nguyên vẹn từ
``code/helper/human_categories.py`` của repository chính thức
https://github.com/rgeirhos/texture-vs-shape . Protocol chính thức lấy trung
bình softmax probability của mọi ImageNet class thuộc mỗi cue category rồi
chọn category có giá trị trung bình lớn nhất; không dùng trực tiếp top-1.
"""

import numpy as np


# Thứ tự alphabet không ảnh hưởng đến argmax, nhưng khớp get_human...() chính thức.
OFFICIAL_16_INDICES = {
    "airplane": [404],
    "bear": [294, 295, 296, 297],
    "bicycle": [444, 671],
    "bird": [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23, 24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145],
    "boat": [472, 554, 625, 814, 914],
    "bottle": [440, 720, 737, 898, 899, 901, 907],
    "car": [436, 511, 817],
    "cat": [281, 282, 283, 284, 285, 286],
    "chair": [423, 559, 765, 857],
    "clock": [409, 530, 892],
    "dog": [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255, 256, 257, 259, 261, 262, 263, 265, 266, 267, 268],
    "elephant": [385, 386],
    "keyboard": [508, 878],
    "knife": [499],
    "oven": [766],
    "truck": [555, 569, 656, 675, 717, 734, 864, 867],
}

CATEGORIES = tuple(sorted(OFFICIAL_16_INDICES))


def decisions_from_probabilities(probabilities: np.ndarray):
    """Trả (decision, confidence) cho tensor/array ``(N, 1000)``.

    ``confidence`` là trung bình probability của category chiến thắng, không
    phải probability top-1 ImageNet. Hàm từ chối logits để tránh áp sai protocol.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 1000:
        raise ValueError("Expected probabilities with shape (N, 1000)")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Expected finite softmax probabilities in [0, 1]")
    category_scores = np.column_stack([
        values[:, OFFICIAL_16_INDICES[category]].mean(axis=1)
        for category in CATEGORIES
    ])
    winners = category_scores.argmax(axis=1)
    return [CATEGORIES[index] for index in winners], category_scores.max(axis=1)
