"""Florence-2 自动标注封装：开放词汇检测 → YOLO 归一化标签。

依赖 transformers 官方集成（Florence-2 为 seq2seq，tokenizer/图像处理器必须
走 transformers，无法纯 ONNX 自研）。坐标换算：
  - Florence-2 OVD 后处理返回"像素坐标"（原始图分辨率），需按图像宽高归一化
    到 [0,1] x,y（在 detect 内完成）；
  - YOLO 标签格式为归一化 (cx, cy, w, h)，全部落在 [0,1] 可直接写入训练。
"""

import logging

import cv2
import numpy as np

from utils.config import DEFAULT_MIN_BOX_SIZE

logger = logging.getLogger(__name__)

# Florence-2 开放词汇检测任务前缀（模型推理文本必须以此开头）
TASK_OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def parse_classes_yaml(path):
    """解析类别定义 yaml，返回 {"names": [...], "prompts": [...]}。

    文件格式（顺序决定 class_id，从 0 开始）：
        names: [cbook, cat]
        prompts: [cbook, cat]      # 可选，默认等于 names
    prompts 用于拼 Florence-2 共享检测提示词；names 用于标签与预览显示。
    缺 prompts 或长度不足时沿用 names。
    """
    import os

    import yaml
    if not os.path.exists(path):
        raise FileNotFoundError(f"类别配置文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    names = list(data.get("names") or [])
    if not names:
        raise ValueError(f"类别配置缺少 names: {path}")
    prompts = list(data.get("prompts") or names)
    if len(prompts) < len(names):
        prompts = names
    return {"names": names, "prompts": prompts}


def build_prompt(class_prompts):
    """把类别提示词列表拼成 Florence-2 共享检测文本（一次推理框出全部类别）。

    Args:
        class_prompts: list[str]，各类别对应的检测提示词。

    Returns:
        完整任务文本，如 "<OPEN_VOCABULARY_DETECTION> cbook. cat. bottle."
    """
    phrase = " ".join(str(p).strip() for p in class_prompts if str(p).strip())
    return f"{TASK_OPEN_VOCABULARY_DETECTION} {phrase}"


def _normalize_boxes(boxes):
    """将 Florence-2 bbox 转为 [0,1] 归一化 (x0,y0,x1,y1)。

    部分实现/版本返回 0-999 整数坐标，这里统一除以 1000；已是 [0,1] 则原样保留。
    """
    out = []
    for b in boxes:
        x0, y0, x1, y1 = (float(v) for v in b)
        if max(x0, y0, x1, y1) > 1.5:
            x0, y0, x1, y1 = x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0
        out.append((x0, y0, x1, y1))
    return out


def boxes_to_yolo(boxes_xyxy):
    """归一化矩框 (x0,y0,x1,y1) 转 YOLO (cx,cy,w,h)，值裁剪到 [0,1]。"""
    boxes = _normalize_boxes(boxes_xyxy)
    records = []
    for x0, y0, x1, y1 in boxes:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        w = x1 - x0
        h = y1 - y0
        records.append(
            (min(max(cx, 0.0), 1.0),
             min(max(cy, 0.0), 1.0),
             min(max(w, 0.0), 1.0),
             min(max(h, 0.0), 1.0))
        )
    return records


def map_label_to_class(labels, prompt_to_id):
    """把 Florence-2 返回的 label 文本映射成 class_id（按提示词匹配）。

    Args:
        labels: list[str]，Florence-2 检测类别标签。
        prompt_to_id: dict[str,int]，提示词 → 类别 id 的映射（键已归一化）。

    Returns:
        list[tuple[int, str]]：每个目标的 (class_id, 原始label)；匹配不到回退 id=0。
    """

    def norm(s):
        return " ".join(str(s).strip().lower().split())

    lookup = {norm(k): v for k, v in prompt_to_id.items()}
    resolved = []
    for lab in labels:
        resolved.append((lookup.get(norm(lab), 0), lab))
    return resolved


def write_yolo_label(txt_path, records, encoding='utf-8'):
    """写入 YOLO 标签 txt，每行 "class_id cx cy w h"（6 位小数）。"""
    lines = []
    for class_id, cx, cy, w, h in records:
        lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(txt_path, 'w', encoding=encoding) as f:
        f.write("\n".join(lines))


def draw_detections(frame_bgr, records, names, box_color=(0, 255, 0), thickness=2):
    """在 BGR 图像上叠加检测框与类别名，返回标注图副本。

    Args:
        records: list[(class_id, (x0,y0,x1,y1), score?)] 或直接 (class_id, box_xyxy)；
                支持带 score 的三元组或二元组。
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for rec in records:
        if len(rec) == 3:
            class_id, (x0, y0, x1, y1), _ = rec
        else:
            class_id, (x0, y0, x1, y1) = rec
        x0 = int(max(0.0, x0) * w) % (w + 1)
        x1 = int(min(1.0, x1) * w) % (w + 1)
        y0 = int(max(0.0, y0) * h) % (h + 1)
        y1 = int(min(1.0, y1) * h) % (h + 1)
        x1 = min(x1, w - 1)
        y1 = min(y1, h - 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), box_color, thickness)
        name = names[class_id] if class_id < len(names) else f"cls{class_id}"
        cv2.putText(img, name, (x0, max(y0 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    return img


class Florence2Annotator:
    """封装 Florence-2 模型/处理器加载与开放词汇检测推理。"""

    # Florence-2 本地仓库必须包含的关键配置文件（缺失时给出下载指引）
    _ESSENTIAL_FILES = {
        "config.json": "模型结构配置",
        "vocab.json": "分词器词表",
    }

    def __init__(self, model_path, device=None, fp16=True):
        import os

        import torch

        self._validate_model_dir(model_path)

        self.device = torch.device('cuda' if (device != 'cpu' and torch.cuda.is_available()) else 'cpu')
        self.fp16 = fp16 and self.device.type == 'cuda'
        dtype = torch.float16 if self.fp16 else torch.float32

        from transformers import AutoModelForCausalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.model_path = model_path
        logger.info(f"Florence-2 模型已加载: {model_path} (device={self.device.type}, fp16={self.fp16})")

    @staticmethod
    def _validate_model_dir(model_path):
        """校验 model_path 是完整的 Florence-2 仓库目录，而非单个权重文件。

        transformers 的 from_pretrained 在路径不存在时会把字符串当 HuggingFace
        仓库名解析（报 "Repo id must use alphanumeric chars..."），这里提前给出中文提示。
        """
        import os

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Florence-2 模型路径必须是完整仓库目录，而不是单个权重文件。\n"
                f"当前得到: {model_path}（该目录不存在）\n"
                f"请把 HuggingFace microsoft/Florence-2-base 的整个仓库下载到该目录，\n"
                f"务必包含 config.json、preprocessor_config.json、vocab.json、\n"
                f"tokenizer_config.json、generation_config.json、model.safetensors 等文件。"
            )

        missing = {name: desc for name, desc in Florence2Annotator._ESSENTIAL_FILES.items()
                   if not os.path.exists(os.path.join(model_path, name))}
        if missing:
            detail = "\n".join(f"  - {name}（{desc}）" for name, desc in missing.items())
            raise FileNotFoundError(
                f"Florence-2 仓库目录 {model_path} 缺少必要配置，无法用 transformers 加载：\n{detail}\n"
                f"请下载完整的 HuggingFace 仓库（当前目录可能只有权重文件），或改用 checkpoint 目录。"
            )

    def detect(self, image_rgb, class_prompts, task=TASK_OPEN_VOCABULARY_DETECTION):
        """对单张 RGB 图像做开放词汇检测。

        Args:
            image_rgb: HxWx3 RGB numpy 数组。
            class_prompts: list[str] 类别提示词。
            task: 任务前缀，默认开放词汇检测。

        Returns:
            list[dict]：{class_id, label, score, box_xyxy(normalized [0,1])}
        """
        import torch

        text = build_prompt(class_prompts)
        inputs = self.processor(text=text, images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.fp16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        prompt_to_id = {p: i for i, p in enumerate(class_prompts)}

        h, w = image_rgb.shape[:2]
        with torch.no_grad():
            generated = self.model.generate(
                inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        result_text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        wrapper = self.processor.post_process_generation(
            result_text, task=task, image_size=(w, h)
        )
        # post_process_generation 返回 {task: {...}} 外层包装，需先解包
        parsed = wrapper.get(task, {}) if isinstance(wrapper, dict) else {}

        labels = parsed.get("labels") or parsed.get("bboxes_labels") or []
        scores = parsed.get("scores") or [1.0] * len(labels)
        bboxes = parsed.get("bboxes") or []
        # OVD 后处理返回像素坐标（原始图分辨率），按图像宽高归一化到 [0,1]
        boxes = [(x0 / w, y0 / h, x1 / w, y1 / h) for x0, y0, x1, y1 in bboxes]
        class_ids = [c for c, _ in map_label_to_class(labels, prompt_to_id)]

        min_size = DEFAULT_MIN_BOX_SIZE
        dets = []
        for cid, lab, sc, (x0, y0, x1, y1) in zip(class_ids, labels, scores, boxes):
            # 过滤 Florence-2 对背景输出的退化点框（边长 ~0 的无效 bbox）
            if (x1 - x0) <= min_size or (y1 - y0) <= min_size:
                continue
            dets.append(
                {"class_id": cid, "label": lab, "score": float(sc), "box_xyxy": (x0, y0, x1, y1)}
            )
        return dets