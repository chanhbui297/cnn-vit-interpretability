"""Fixed baseline models; evaluation mode does not disable autograd."""

import torch
import torchvision

# Dùng lại khai báo của baseline để tránh chọn nhầm checkpoint giữa các thí nghiệm.
from .evaluate_baseline import MODEL_SPECS


def load_model(model_name, device=None, *, allow_download=False):
    """Return (model, official_transform, metadata), reusing baseline weights.

    Downloads are opt-in. Parameters remain trainable for attribution methods.
    Call from the repository root with ``from src.models import load_model``.
    """
    # Tra tên model để lấy hàm khởi tạo và bộ weights đã cố định trong baseline.
    specs = {name: (factory, weights) for name, factory, weights in MODEL_SPECS}
    if model_name not in specs:
        raise ValueError(f"Unknown model {model_name!r}; choose {list(specs)}")
    factory, weights = specs[model_name]
    checkpoint = weights.url.rsplit("/", 1)[-1]
    # Mặc định chỉ đọc weights đã có trên máy; muốn tải phải truyền allow_download=True.
    if not allow_download:
        from pathlib import Path

        cached = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint
        if not cached.is_file():
            raise FileNotFoundError(
                f"Missing cached weights: {cached}. Set allow_download=True to download."
            )
    # Ưu tiên device do người gọi chọn; nếu bỏ trống thì tự chọn CUDA hoặc CPU.
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Mỗi checkpoint có transform riêng; không thay bằng preprocessing tự viết.
    transform = weights.transforms()
    model = factory(weights=weights).to(device)
    # eval() tắt dropout/cập nhật BatchNorm nhưng vẫn cho phép tính gradient.
    # Không bọc model trong inference_mode vì Grad-CAM/ERF còn cần autograd.
    model.eval()
    # Metadata giúp ghi lại weights, phiên bản và preprocessing khi bàn giao kết quả.
    metadata = {
        "model_name": model_name,
        "weights": str(weights),
        "weights_url": weights.url,
        "checkpoint": checkpoint,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "preprocessing": {
            "color_mode": "RGB",
            "resize_size": transform.resize_size,
            "crop_size": transform.crop_size,
            "interpolation": transform.interpolation.value,
            "antialias": transform.antialias,
            "mean": transform.mean,
            "std": transform.std,
        },
    }
    # Người gọi tự chạy thí nghiệm; hàm này không dự đoán hoặc ghi file kết quả.
    return model, transform, metadata
