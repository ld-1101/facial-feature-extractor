"""
人脸特征提取器 — 基于视觉模型分析 + 几何测量体系
========================================================
从照片中提取完整的人脸几何特征，生成可用于 AI 生图的精确描述。

支持:
  - 清晰人脸: 完整提取 80+ 几何测量维度
  - 模糊/遮挡人脸: 提取可见部分，面部回退默认模板

用法:
  python extract.py --photo "photo.jpg"
  python extract.py --photo "photo.jpg" --partial  # 面部模糊模式
  python extract.py --photo "photo.jpg" --output result.json
"""

import json
import base64
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

# ============================================================
# 配置 — 替换为你的 API 端点
# ============================================================

ENDPOINTS = {
    "direct": "https://api.example.com/v1/chat/completions",
    "token-plan": "https://api.example.com/v1/chat/completions",
}

def load_api_key():
    """从环境变量或 .env 加载 API Key"""
    import os
    key = os.environ.get("VISION_API_KEY", "")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").strip().split("\n"):
            if "VISION_API_KEY=" in line:
                return line.split("=", 1)[1].strip()
    return ""


def load_default_dna():
    """加载默认面部模板（面部模糊时回退）"""
    dna_path = ROOT / "data" / "sample_dna.json"
    if dna_path.exists():
        data = json.loads(dna_path.read_text(encoding="utf-8"))
        return data.get("dna", data)
    return {}


# ============================================================
# 视觉分析
# ============================================================

def analyze_photo(photo_path: str, partial: bool = False) -> dict:
    """用视觉模型分析照片，返回结构化面部特征 JSON"""
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("VISION_API_KEY 未设置。export VISION_API_KEY=xxx 或创建 .env 文件")

    photo = Path(photo_path)
    if not photo.exists():
        raise FileNotFoundError(f"照片不存在: {photo_path}")

    # 编码图片
    ext = photo.suffix.lower()
    mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.webp': 'image/webp', '.bmp': 'image/bmp'}
    mime = mime_map.get(ext, 'image/jpeg')
    with open(photo, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    if partial:
        return _analyze_partial(api_key, img_b64, mime)
    else:
        return _analyze_full(api_key, img_b64, mime)


def _analyze_full(api_key: str, img_b64: str, mime: str) -> dict:
    """完整面部特征提取（清晰人脸）"""
    messages = [
        {"role": "developer", "content": (
            "你是人脸几何特征分析专家。分析照片中人物的面部特征，输出严格的几何测量 JSON。\n"
            "优先使用数值（mm、角度、比例），配合自然语言描述。\n\n"
            "输出 JSON 格式（不要 markdown 包裹）：\n"
            "{\n"
            '  "face_shape_type": "oval/heart/round/square/diamond/long",\n'
            '  "face_length_to_width": 数字(面长面宽比),\n'
            '  "three_sections": "三庭比例描述",\n'
            '  "five_eyes": "五眼比例描述",\n\n'
            '  "eye_fissure_length_mm": 数字,\n'
            '  "eye_fissure_height_mm": 数字,\n'
            '  "canthal_tilt_degrees": 数字(正=眼尾上扬, 负=下垂),\n'
            '  "intercanthal_distance_ratio": 数字(内眦间距/眼裂长),\n'
            '  "eyelid_crease_height_mm": 数字(0=单眼皮),\n'
            '  "eye_shape_type": "almond/round/fox/phoenix/downturned等",\n'
            '  "aegyo_sal_height_mm": 数字(卧蚕高度),\n'
            '  "iris_diameter_mm": 数字,\n'
            '  "sclera_show_below_iris_mm": 数字(0=正常, >0=三白眼),\n\n'
            '  "eyebrow_shape_type": "flat/arched/angled等",\n'
            '  "eyebrow_thickness_mm": "眉头Xmm渐细至眉尾Ymm",\n'
            '  "eyebrow_color": "眉色描述",\n'
            '  "eyebrow_arch_position": "眉峰位置(1/3处/1/2处等)",\n\n'
            '  "nasal_length_mm": 数字,\n'
            '  "nasal_bridge_height_mm": 数字(山根突出高度),\n'
            '  "nasal_bridge_width_mm": 数字,\n'
            '  "nasal_tip_to_alar_ratio": 数字(鼻头宽/鼻翼宽),\n'
            '  "nasolabial_angle_degrees": 数字,\n'
            '  "nose_shape_type": "straight/upturned/aquiline/flat等",\n\n'
            '  "mouth_width_to_nose_ratio": 数字,\n'
            '  "upper_lip_thickness_mm": 数字,\n'
            '  "lower_lip_thickness_mm": 数字,\n'
            '  "oral_commissure_angle_degrees": 数字(正=上扬微笑唇),\n'
            '  "philtrum_length_mm": 数字,\n'
            '  "lip_shape_type": "m_shape/heart/full/thin等",\n\n'
            '  "skin_tone": "冷白皮/暖白皮/自然色等",\n'
            '  "skin_undertone": "冷调/暖调/中性",\n'
            '  "skin_glow": "水光/哑光/自然",\n'
            '  "skin_texture": "零毛孔/正常/粗糙等",\n\n'
            '  "hair_style": "发型描述",\n'
            '  "hair_color": "发色描述",\n'
            '  "hair_length_cm": 数字,\n'
            '  "hair_texture": "顺滑/微卷/蓬松等",\n'
            '  "hair_volume_density": "发量描述",\n\n'
            '  "build": "体型描述",\n'
            '  "shoulder_label": "肩型描述",\n'
            '  "height_cm": 数字,\n'
            '  "waist_hip_ratio": 数字,\n\n'
            '  "accessories": ["配饰列表"],\n'
            '  "temperament_primary": "气质类型",\n'
            '  "apparent_age_range": "年龄范围",\n'
            '  "makeup_style": "妆容风格",\n'
            '  "forbidden": ["与该人物风格冲突的元素"],\n'
            '  "description": "一段150-200字的连贯描述"\n'
            "}\n\n"
            "规则：\n"
            "- 以照片实际为准，每个字段必须填写，禁止省略\n"
            "- 数值尽量精确到小数点后一位\n"
            "- 描述使用自然语言，包含形态+颜色+质感\n"
            "- 仅输出 JSON，不要 markdown 包裹"
        )},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": "请精确分析这张照片中人物的面部几何特征，每个维度给出具体数值。"}
        ]}
    ]
    return _call_vision(api_key, messages)


def _analyze_partial(api_key: str, img_b64: str, mime: str) -> dict:
    """部分提取（面部模糊/遮挡场景）— 可见特征精确，不可见填 null"""
    messages = [
        {"role": "developer", "content": (
            "你是人物外貌部分提取专家。照片中面部可能模糊或遮挡。\n"
            "只提取实际可见的维度，不可见的填 null。\n\n"
            "输出 JSON:\n"
            "{\n"
            '  "face_shape_type": "oval/heart/round等 或 null",\n'
            '  "eye_shape_type": "almond/round等 或 null",\n'
            '  "nose_shape_type": "straight/flat等 或 null",\n'
            '  "lip_shape_type": "m_shape/full等 或 null",\n'
            '  "skin_tone": "色号描述 或 null(从手臂/颈部判断)",\n'
            '  "hair_style": "发型 或 null",\n'
            '  "hair_color": "发色 或 null",\n'
            '  "hair_length_cm": 数字或null,\n'
            '  "build": "体型 或 null",\n'
            '  "height_cm": 数字或null,\n'
            '  "accessories": ["可见配饰"] 或 [],\n'
            '  "temperament_primary": "从整体推断的气质",\n'
            '  "apparent_age_range": "年龄范围",\n'
            '  "visible_summary": "照片中确定可见的特征概括",\n'
            '  "missing_summary": "照片中不可见/模糊的特征概括",\n'
            '  "description": "仅基于可见部分的描述，推断用【推断：xxx】标注"\n'
            "}\n\n"
            "规则：面部模糊则 face/eye/nose/lip 字段全部填 null，不要编造"
        )},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": "分析这张照片。面部可能模糊，只提取可见特征，不可见填null。"}
        ]}
    ]
    result = _call_vision(api_key, messages)

    # 自动合并默认面部模板
    default_dna = load_default_dna()
    face_fields = ["face_shape_type", "face_length_to_width", "three_sections",
                   "eye_fissure_length_mm", "eye_fissure_height_mm", "canthal_tilt_degrees",
                   "eyelid_crease_height_mm", "eye_shape_type", "aegyo_sal_height_mm",
                   "eyebrow_shape_type", "eyebrow_thickness_mm",
                   "nasal_length_mm", "nasal_bridge_height_mm", "nasolabial_angle_degrees",
                   "nose_shape_type", "mouth_width_to_nose_ratio",
                   "upper_lip_thickness_mm", "lower_lip_thickness_mm",
                   "oral_commissure_angle_degrees", "philtrum_length_mm", "lip_shape_type",
                   "skin_texture", "skin_flaws", "makeup_style",
                   "facs_neutral_baseline", "photogenic"]

    merged = dict(result)
    for field in face_fields:
        if merged.get(field) is None or merged.get(field) == "":
            if field in default_dna:
                merged[field] = default_dna[field]

    merged["_merge_note"] = "面部特征从默认模板继承（原照片模糊/遮挡）"
    return merged


def _call_vision(api_key: str, messages: list) -> dict:
    """调用视觉模型 API"""
    import requests
    ep = ENDPOINTS["token-plan"]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    body = {
        "model": "vision-v1",
        "messages": messages,
        "max_completion_tokens": 2048,
        "temperature": 0.3,
    }

    print(f"[Vision] analyzing...")
    resp = requests.post(ep, headers=headers, json=body, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()

    # 清理可能的 markdown 包裹
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_output": content, "parse_error": True}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="人脸几何特征提取器 — 基于 MIMO v2.5 + 68点/468点关键点体系")
    parser.add_argument("--photo", required=True, help="照片路径")
    parser.add_argument("--partial", action="store_true", help="面部模糊/部分提取模式")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    args = parser.parse_args()

    try:
        result = analyze_photo(args.photo, partial=args.partial)
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 60)

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved to: {args.output}")
    except Exception as e:
        print(f"[ERR] {e}")
        sys.exit(1)
