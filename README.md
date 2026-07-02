# Facial Feature Extractor

基于**视觉模型** + **人脸关键点几何测量体系**（参照 dlib 68点 / MediaPipe 468点 / Farkas 颅面人体测量学）的人脸特征提取器。

从照片中自动提取 **80+ 个可量化的人脸几何维度**，生成可供 AI 生图模型（Seedream / Stable Diffusion / Midjourney）直接使用的精确面部描述。

## ✨ 核心能力

| 功能 | 说明 |
|------|------|
| 📐 **几何测量** | 眼裂长度(mm)、眦角倾斜度(°)、鼻唇角(°)、面长面宽比等 |
| 🔍 **双模式** | 清晰人脸完整提取 / 模糊人脸部分提取 + 模板补齐 |
| 📝 **提示词生成** | DNA JSON → Seedream/Stable Diffusion 可直接使用的 prompt |
| 📚 **特征库** | 内置 84 维语义特征枚举 + 黄金比例参考 |

## 🚀 快速开始

### 1. 安装
```bash
pip install requests
```

### 2. 配置 API

设置环境变量或创建 `.env` 文件：
```bash
export VISION_API_KEY=your-api-key
# 或 .env: VISION_API_KEY=your-api-key
```

> 支持任何兼容 OpenAI Chat Completions 格式的视觉模型 API。替换 `extract.py` 中的 `ENDPOINTS` 和 `model` 名称即可。

### 3. 分析照片
```bash
# 清晰人脸 — 完整提取
python extract.py --photo my_photo.jpg

# 模糊人脸 — 部分提取 + 默认模板补齐
python extract.py --photo blurred_face.jpg --partial

# 保存结果
python extract.py --photo my_photo.jpg --output result.json
```

### 4. 生成生图提示词
```bash
# 半身肖像
python generate_prompt.py --dna result.json --type portrait

# 全身展示
python generate_prompt.py --dna result.json --type fullbody
```

## 📊 输出示例

```json
{
  "face_shape_type": "oval",
  "face_length_to_width": 1.42,
  "three_sections": "三庭均等 1:1:1",
  "eye_fissure_length_mm": 30,
  "eye_fissure_height_mm": 11,
  "canthal_tilt_degrees": 6,
  "eyelid_crease_height_mm": 4.5,
  "nasal_bridge_height_mm": 6.5,
  "nasolabial_angle_degrees": 100,
  "skin_tone": "冷白皮(偏粉调)",
  "hair_color_hex": "#4A3629",
  "height_cm": 168,
  "waist_hip_ratio": 0.68,
  "temperament_primary": "东方初恋脸",
  "apparent_age_range": "22-26岁"
}
```

## 📐 几何测量体系

- **dlib 68-point** — iBUG 300-W 数据集标准
- **MediaPipe 468-point** — 3D 面部网格
- **Farkas 颅面人体测量学** — 学术标准
- **面部黄金比例** — 1:1.618

详见 `data/facial_landmark_system.json`。

## 📁 文件结构

```
facial-feature-extractor/
├── extract.py                       # 主提取脚本
├── generate_prompt.py               # 提示词生成器
├── data/
│   ├── facial_landmark_system.json  # 几何测量框架
│   ├── facial_features_library.json # 语义特征枚举库
│   └── sample_dna.json              # 示例 DNA 模板
├── requirements.txt
└── README.md
```

## 📄 License

MIT

## License

Private repository. All rights reserved.
