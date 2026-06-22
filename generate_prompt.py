"""
从 DNA JSON 生成 AI 生图提示词
=============================
将几何测量值自动转换为 Seedream / Midjourney / SD 可用的提示词。

用法:
  python generate_prompt.py --dna data/sample_dna.json
  python generate_prompt.py --dna result.json --type portrait
  python generate_prompt.py --dna result.json --type fullbody
"""

import json
import argparse
from pathlib import Path


def generate(dna: dict, prompt_type: str = "portrait") -> str:
    """从 DNA 生成提示词"""
    def g(*keys):
        for k in keys:
            v = dna.get(k, '')
            if v: return v
        return ''

    age = str(dna.get('apparent_age_range', '22-26')).rstrip('岁')
    height = dna.get('height_cm', '168')

    # 面部
    face = g('face_shape_label', 'face_shape_type') or '标准鹅蛋脸'
    eyes = g('eye_shape_label') or (
        f"{g('eye_shape_type', 'eyes')} — "
        f"眼裂{g('eye_fissure_length_mm', '30')}×{g('eye_fissure_height_mm', '11')}mm, "
        f"眦角+{g('canthal_tilt_degrees', '6')}°"
    )
    brows = g('eyebrow_label', 'eyebrow_shape_type') or '自然平眉'
    nose = g('nose_label') or (
        f"{g('nose_shape_type', 'nose')} — "
        f"鼻长{g('nasal_length_mm', '46')}mm, 山根{g('nasal_bridge_height_mm', '6.5')}mm"
    )
    lips = g('lip_label') or (
        f"{g('lip_shape_type', 'lips')} — "
        f"上{g('upper_lip_thickness_mm', '7')}mm下{g('lower_lip_thickness_mm', '10')}mm"
    )
    skin = g('skin_label', 'skin_tone') or '冷白皮水光肌'
    hair = g('hair_label', 'hair_style') or '及腰长直发'
    makeup = g('makeup_label', 'makeup_style') or '韩式自然裸妆'
    temper = g('temperament_primary', 'style_keyword') or '温柔甜美'

    # 配饰
    acc_list = dna.get('accessories', [])
    acc = '。佩戴' + '、'.join(acc_list) if acc_list else ''

    # 体型
    build = g('build_label', 'build') or '纤瘦匀称'

    # Negative prompt
    forbidden = dna.get('forbidden', [])
    neg_parts = [x for x in forbidden[:12] if len(x) <= 20]
    neg = 'NSFW, nude, ' + ', '.join(neg_parts)
    neg += ', low quality, blurry, distorted face, extra limbs, watermark, text'

    if prompt_type == "portrait":
        main = (
            f"超写实人像摄影, 竖屏9:16, 柔光拍摄, 中景半身肖像, 白色简洁背景。"
            f"{age}岁亚洲女性。{face}。{eyes}。{brows}。{nose}。{lips}。"
            f"{skin}。{hair}。{acc}。{makeup}。气质{temper}。"
            f"高清画质, 皮肤质感真实, 8K, 摄影棚柔光, 佳能R5 85mm f/1.2。"
        )
    elif prompt_type == "fullbody":
        main = (
            f"超写实时尚摄影, 竖屏9:16, 全身展示, 白色背景, 电商服装拍摄风格。"
            f"{age}岁亚洲女性, 身高{height}cm, {build}。{face}。{eyes}。"
            f"{nose}。{lips}。{skin}。{hair}。{acc}。{makeup}。"
            f"气质{temper}。正面站立, 双手自然垂放, 微笑看向镜头。"
            f"高清画质, 8K, 摄影棚均匀柔光, 佳能R5 50mm f/1.4。"
        )
    else:
        raise ValueError(f"Unknown type: {prompt_type}")

    return f"{main}\n\nNegative prompt: {neg}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 DNA JSON 生成 AI 生图提示词")
    parser.add_argument("--dna", required=True, help="DNA JSON 文件路径")
    parser.add_argument("--type", default="portrait",
                        choices=["portrait", "fullbody"], help="提示词类型")
    args = parser.parse_args()

    dna = json.loads(Path(args.dna).read_text(encoding="utf-8"))
    # 如果 DNA 嵌套在顶层 key 下
    if "dna" in dna:
        dna = dna["dna"]

    prompt = generate(dna, args.type)
    print(prompt)
