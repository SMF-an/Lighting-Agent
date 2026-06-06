# 🎨 基于大模型微调与物理色域约束的高保真光效智能体

> High-Fidelity Lighting Effect Agent via Fine-Tuned Diffusion Model with Physical Gamut Constraints

一个端到端的光效生成流水线：将**自然语言场景描述** → **结构化光效提示词** → **微调后的 Stable Diffusion 生成渐变光效图** → **Lab 空间误差扩散抖动映射到物理 LED 色域**。

---

## 📋 目录

- [项目概览](#项目概览)
- [流水线架构](#流水线架构)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细用法](#详细用法)
  - [1. 数据准备](#1-数据准备)
  - [2. 模型微调](#2-模型微调)
  - [3. 推理生成](#3-推理生成)
  - [4. Web 演示](#4-web-演示)
  - [5. 评估](#5-评估)
  - [6. 消融实验](#6-消融实验)
- [如何进一步微调](#如何进一步微调)
- [目录结构](#目录结构)
- [已训练模型](#已训练模型)
- [评估结果](#评估结果)
- [核心创新点](#核心创新点)
- [引用](#引用)

---

## 项目概览

本项目针对"根据场景描述生成可物理部署的高保真光效"这一任务，设计了一条完整的生成流水线。核心思路是：

1. **LLM 语义翻译** — 利用大语言模型将中/英文场景描述翻译为结构化的光效提示词，排除"室内场景"语义泄露
2. **扩散模型生成** — 在程序化生成的光效数据集上微调 Stable Diffusion v1.5，使其专注于生成渐变光效图像
3. **物理色域约束** — 通过 CIE Lab 空间中的 Floyd-Steinberg 误差扩散抖动算法，将生成图像映射到物理 LED 灯具的 SDL 色域范围内

### 关键特性

- ✅ **完整的训练流程** — 支持完整微调（Full Fine-tuning）和 LoRA 两种模式
- ✅ **物理色域约束** — Lab 空间 Floyd-Steinberg 抖动，消除条带伪影
- ✅ **多种损失函数** — MSE + 色彩损失（CIELab 色度/饱和度约束）+ 高频损失（频域正则化）
- ✅ **场景防泄露** — 30+ 禁止词汇过滤 + JSON 结构化约束 + 强制后缀
- ✅ **Web 演示** — 基于 Gradio 的交互式 Web 界面
- ✅ **综合评估框架** — CLIP 分数、KID、场景泄露、高频能量、提示一致性等多维指标
- ✅ **一键消融实验** — 对比不同超参数（步数、引导尺度、负向提示词）的影响

---

## 流水线架构

```
┌──────────────────────────────────────────────────────────┐
│                    完整推理流水线                           │
├───────────┬───────────┬───────────┬──────────────────────┤
│  模块 1   │  模块 2   │  模块 3   │       模块 4          │
│  LLM 翻译  │ SD 生成   │ 候选选择   │   SDL 色域映射        │
│           │           │ (可选)    │                      │
├───────────┼───────────┼───────────┼──────────────────────┤
│ 场景描述   │ 光效提示词  │ 原始图像   │   物理合规图像        │
│ "酒店大堂"  │ → prompt → │ 512×512  │ → 色域内 LED 图像     │
└───────────┴───────────┴───────────┴──────────────────────┘
```

**训练流水线**：

```
程序化生成光效图像 (2000张)
  ├── simulate_graph_one.py (1000张, 双色线性渐变)
  └── simulate_graph_more.py (1000张, 多色流体光晕)
       │
       ▼
VLM 图像标注 (Qwen-VL-Max)
  └── 结构化 JSON 标题 + 场景词汇过滤
       │
       ▼
微调 Stable Diffusion v1.5
  ├── full_finetune.py (完整微调: MSE + Color Loss + High-Freq Loss)
  └── lora_finetune.py (LoRA: 仅 MSE)
```

---

## 环境要求

- **Python** ≥ 3.10
- **CUDA** ≥ 11.8（推荐 12.1+）
- **GPU 显存** ≥ 12 GB（训练），≥ 8 GB（推理）
- **磁盘空间** ≥ 20 GB（含数据集和检查点）

### 核心依赖

| 包名 | 用途 |
|------|------|
| `torch` ≥ 2.0 | 深度学习框架 |
| `diffusers` ≥ 0.25 | Stable Diffusion 流水线 |
| `transformers` | CLIP 文本编码器与评估 |
| `peft` | LoRA 微调 |
| `datasets` | Hugging Face 数据集加载 |
| `gradio` ≥ 4.0 | Web 演示界面 |
| `Pillow`, `numpy`, `scipy` | 图像处理 |
| `matplotlib` | 可视化 |
| `openai` ≥ 1.0 | LLM/VLM API 调用 |
| `python-dotenv` | 环境变量管理 |
| `torch-fidelity` | KID 指标计算 |
| `python-pptx` | 幻灯片生成（可选） |

---

## 安装

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd final_lab

# 2. 创建虚拟环境（推荐使用 uv）
uv sync
# 或使用 pip
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 环境变量配置（`.env`）

```bash
# LLM API（用于提示词翻译，模块 1）
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_ID=deepseek-v4-pro

# VLM API（用于数据标注和评估）
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VLM_MODEL_ID=qwen-vl-max-latest

# Hugging Face（用于上传模型/数据集）
HF_TOKEN=hf_your_token_here
```

---

## 快速开始

最简单的使用方式：直接运行 Web 演示，输入场景描述即可生成光效图。

```bash
# 启动 Gradio Web 界面
python src/infer/web_app.py
```

然后在浏览器中打开 `http://localhost:7860`，输入场景描述（如 "温馨的咖啡厅"），点击生成。

### 命令行推理

```bash
# 单次推理
python src/infer/pipeline_infer.py "酒店大堂，金色暖色调" \
    --checkpoint-type full \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --sdl-file data/SDL2_0.txt \
    --output-dir output/
```

---

## 详细用法

### 1. 数据准备

#### 1.1 生成光效图像

```bash
# 完整数据集构建流程（生成 2000 张图像 + VLM 标注）
python src/data/build_dataset_pipeline.py \
    --root-dir data \
    --num-images-one 1000 \
    --num-images-more 1000 \
    --width 768 \
    --height 768 \
    --caption-mode controlled
```

参数说明：
- `--num-images-one`：双色线性渐变图像数量（默认 1000）
- `--num-images-more`：多色流体光晕图像数量（默认 1000）
- `--caption-mode`：标注模式（`controlled` 结构化 JSON / `free` 自由形式）
- `--seed-one` / `--seed-more`：随机种子（默认 42 / 43）

#### 1.2 单独生成图像（不标注）

```bash
# 生成双色渐变
python src/data/simulate_graph_one.py

# 生成多色流体光晕
python src/data/simulate_graph_more.py
```

#### 1.3 单独标注图像

```bash
# 使用 VLM 为已有图像生成标题
python src/data/generate_light_caption.py \
    --image-dir data/images \
    --mode controlled \
    --output data/light_effect_captions.jsonl
```

#### 1.4 随机提示词（不依赖 VLM）

```bash
# 从词汇表中随机采样提示词
python src/data/generate_random_prompt.py \
    --num-prompts 1000 \
    --output data/random_prompts.jsonl
```

---

### 2. 模型微调

#### 2.1 完整微调（Full Fine-tuning）

微调 UNet 全部参数，可选联合微调文本编码器。

```bash
# 基础微调（仅 MSE 损失）
python src/finetune/full_finetune.py \
    --base-model runwayml/stable-diffusion-v1-5 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-5

# 完整微调（MSE + 色彩损失 + 高频损失 + 文本编码器）
python src/finetune/full_finetune.py \
    --base-model runwayml/stable-diffusion-v1-5 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --text-encoder-learning-rate 1e-6 \
    --use-color-loss \
    --color-loss-weight 0.005 \
    --use-highfreq-loss \
    --highfreq-weight 10 \
    --highfreq-cutoff 0.03 \
    --caption-dropout 0.1 \
    --sample-every-steps 2000 \
    --checkpoint-every-steps 4000
```

完整微调关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-model` | `runwayml/stable-diffusion-v1-5` | 基座模型 |
| `--num-epochs` | 10 | 训练轮数 |
| `--batch-size` | 1 | 批次大小（受限于显存） |
| `--learning-rate` | 1e-5 | UNet 学习率 |
| `--text-encoder-learning-rate` | 1e-6 | 文本编码器学习率 |
| `--use-color-loss` | False | 是否启用色彩损失 |
| `--color-loss-weight` | 0.005 | 色彩损失权重 |
| `--use-highfreq-loss` | False | 是否启用高频损失 |
| `--highfreq-weight` | 10 | 高频损失权重 |
| `--highfreq-cutoff` | 0.03 | 频域截止频率 |
| `--caption-dropout` | 0.0 | 标题随机丢弃概率（防止过拟合） |
| `--no-text-encoder` | False | 不训练文本编码器 |
| `--resume-from-checkpoint` | — | 从检查点恢复训练 |

#### 2.2 LoRA 微调

仅训练 UNet 注意力层的低秩适配矩阵，显存占用更低，训练更快。

```bash
python src/finetune/lora_finetune.py \
    --base-model runwayml/stable-diffusion-v1-5 \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --lora-rank 32 \
    --lora-alpha 32 \
    --lora-dropout 0.1
```

LoRA 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora-rank` | 32 | LoRA 秩（越大表达能力越强，但参数量增加） |
| `--lora-alpha` | 32 | LoRA 缩放因子 |
| `--lora-dropout` | 0.1 | LoRA dropout 率 |
| `--learning-rate` | 1e-4 | 学习率（LoRA 通常比完整微调大） |

#### 2.3 损失函数详解

**色彩损失（Color Loss）**：将预测的 x₀ 通过 VAE 解码为 RGB，再转换到 CIELab 色彩空间：
- **色度惩罚**：惩罚色度 < 18 的像素，防止生成"灰蒙蒙"的图
- **饱和度惩罚**：当饱和度 < 0.20 的像素超过 15% 时施加惩罚

**高频损失（High-Freq Loss）**：对亮度通道进行 rFFT 变换，构建十字形掩码（水平/垂直方向低频保留，高频抑制），惩罚掩码外的高频能量占比，促使模型生成更平滑的渐变过渡。

---

### 3. 推理生成

#### 3.1 完整流水线推理

```bash
python src/infer/pipeline_infer.py "温馨的咖啡厅，暖黄灯光" \
    --checkpoint-type full \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --sdl-file data/SDL2_0.txt \
    --output-dir output/ \
    --num-steps 30 \
    --guidance-scale 7.5 \
    --sampler dpm \
    --use-candidate-selection \
    --num-candidates 4 \
    --enable-vlm-eval
```

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scene` | (必填) | 场景描述文本（中/英文均可） |
| `--checkpoint-type` | `full` | 检查点类型：`pretrain` / `full` / `lora` |
| `--checkpoint-dir` | — | 检查点目录路径 |
| `--sdl-file` | `data/SDL2_0.txt` | SDL 色域边界文件 |
| `--num-steps` | 30 | 去噪步数 |
| `--guidance-scale` | 7.5 | 无分类器引导尺度 |
| `--sampler` | `dpm` | 采样器：`ddim` / `dpm` / `dpmpp` / `euler` / `heun` / `unipc` |
| `--seed` | 随机 | 随机种子 |
| `--use-candidate-selection` | False | 生成多个候选，由 VLM 选择最佳 |
| `--num-candidates` | 4 | 候选数量 |
| `--enable-vlm-eval` | False | 启用 VLM 评估生成质量 |
| `--return-process` | False | 返回中间去噪过程图 |

#### 3.2 仅运行图像生成（不含流水线）

```bash
python src/infer/inference.py \
    --checkpoint-type full \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --prompt "Abstract light effect, warm amber to cream gradient, cozy atmosphere" \
    --negative-prompt "furniture, room, objects" \
    --num-steps 50 \
    --guidance-scale 7.5 \
    --sampler dpm \
    --seed 42 \
    --return-process
```

#### 3.3 仅运行 SDL 色域映射

```bash
python src/infer/sdl_converter.py \
    --input image.png \
    --sdl-file data/SDL2_0.txt \
    --output output_gamut.png
```

---

### 4. Web 演示

```bash
python src/infer/web_app.py
```

功能：
- 🎛️ 调整采样参数（步数、引导尺度、采样器、随机种子）
- 🖼️ 查看生成候选图和中间去噪过程
- 📝 查看 LLM 翻译后的结构化提示词
- 🔄 并排对比原始生成图与 SDL 色域映射后的图像
- 📊 查看 VLM 评分结果

---

### 5. 评估

#### 5.1 运行完整评估

```bash
python src/evaluation.py \
    --checkpoint-type full \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --num-prompts 16 \
    --samples-per-prompt 8 \
    --num-steps 30 \
    --guidance-scale 7.5 \
    --output-dir result/eval/my_run/
```

#### 5.2 评估指标说明

| 指标 | 方向 | 含义 |
|------|------|------|
| **CLIP Score** | ↑ 越高越好 | 生成图像与提示词的语义对齐程度 |
| **KID** (Kernel Inception Distance) | ↓ 越低越好 | 生成分布与参考分布的距离 |
| **Prompt Consistency** | ↑ 越高越好 | 同一提示词下多张图像的风格一致性 |
| **Scene Leakage** | ↓ 越低越好 | 图像与室内场景概念的最大 CLIP 相似度（越低说明越"纯光效"） |
| **High-Freq Energy** | ↓ 越低越好 | 高频噪声能量（越低说明渐变越平滑） |
| **Avg Saturation** | — | 平均色彩饱和度 |

#### 5.3 训练集指标统计

```bash
python src/compute_trainset_metrics.py
```

---

### 6. 消融实验

```bash
# 对比不同引导尺度
python src/experiment.py \
    --experiment-type guidance_scale \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --output-dir result/experiment/guidance_scale/

# 对比不同去噪步数
python src/experiment.py \
    --experiment-type num_steps \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --output-dir result/experiment/num_steps/

# 对比有无负向提示词
python src/experiment.py \
    --experiment-type negative_prompt \
    --checkpoint-dir runs/full_color_high_freq_text/checkpoints/final \
    --output-dir result/experiment/negative_prompt/
```

---

## 如何进一步微调

如果你想在新的数据或场景上进一步微调模型，可以按以下步骤操作。

### 方式一：调整数据分布

当前数据集包含 2000 张程序化生成的光效图像。你可以：

```bash
# 1. 修改图像生成参数，调整光效风格
# 编辑 src/data/simulate_graph_more.py 中的调色板、混合权重等参数
# 例如调整色彩和谐模型的比例：
#   HARMONY_MIX = {"analogous": 0.5, "split_complementary": 0.3, "soft_triadic": 0.2}

# 2. 重新生成图像
python src/data/simulate_graph_more.py \
    --num-images 500 \
    --seed 999 \
    --output-dir data/images_new/

# 3. 用 VLM 标注新图像
python src/data/generate_light_caption.py \
    --image-dir data/images_new/ \
    --mode controlled \
    --output data/new_captions.jsonl

# 4. 合并数据集后重新训练
```

### 方式二：在新的基座模型上微调

```bash
# 使用 SDXL 或其他变体作为基座模型
python src/finetune/full_finetune.py \
    --base-model stabilityai/stable-diffusion-xl-base-1.0 \
    --num-epochs 10 \
    --learning-rate 5e-6 \
    --use-color-loss \
    --use-highfreq-loss \
    --output-dir runs/sdxl_experiment/
```

> **注意**：更换基座模型需要确认 diffusers 兼容性和 VAE 编解码逻辑。当前色彩损失依赖于 SD1.5 的 VAE，如果更换模型需要同步调整。

### 方式三：调整损失函数权重

在 `src/finetune/full_finetune.py` 中，你可以通过命令行参数或直接修改代码来调整损失权重：

```bash
# 增大色彩损失权重（让颜色更鲜艳）
python src/finetune/full_finetune.py \
    --use-color-loss \
    --color-loss-weight 0.01 \    # 默认 0.005，增大使颜色更强
    --use-highfreq-loss \
    --highfreq-weight 20 \         # 默认 10，增大使渐变更平滑
    --highfreq-cutoff 0.05         # 默认 0.03，增大保留更多高频细节
```

### 方式四：修改色彩损失逻辑

编辑 `src/finetune/full_finetune.py` 中的 `color_loss` 函数：

```python
# 调整色度阈值 — 控制"灰色容忍度"
chroma_mask = (chroma < 18).float()   # 默认 18，增大则更严格

# 调整饱和度阈值 — 控制低饱和度惩罚
low_sat_mask = (saturation < 0.20).float()  # 默认 0.20
low_sat_ratio_threshold = 0.15              # 默认 0.15

# 替换色彩空间（如改用 HSV 或 OKLab）
# 编辑 RGB→色彩空间 的转换逻辑
```

### 方式五：更换 LLM/VLM

在 `src/infer/prompt_translator.py` 中，LLM 通过 OpenAI 兼容 API 调用：

```bash
# 使用其他兼容 OpenAI API 的模型
export LLM_BASE_URL=https://your-api-endpoint/v1
export LLM_API_KEY=your-key
export LLM_MODEL_ID=your-model-name

# 如使用本地部署的 Qwen2.5 或 DeepSeek
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL_ID=qwen2.5-72b
```

### 方式六：添加新的提示词模板

编辑 `src/infer/prompt_translator.py` 中的场景-调色板知识库：

```python
SCENE_PALETTE_KB = {
    # 添加新场景类别
    "博物馆": {
        "hue_range": "warm neutral (amber, cream, soft beige)",
        "temperature": 3500,
        "mood": "solemn, focused, reverent",
        "example": "Abstract light effect, gradient from warm amber to soft beige, ..."
    },
    # ... 已有类别：教育、餐厅、酒吧、酒店大堂
}
```

### 方式七：从现有检查点继续训练

```bash
# 从已有检查点恢复训练
python src/finetune/full_finetune.py \
    --resume-from-checkpoint runs/full_color_high_freq_text/checkpoints/checkpoint-8000 \
    --num-epochs 5 \
    --learning-rate 5e-6    # 使用更小的学习率
```

### 方式八：调整 SDL 色域映射参数

编辑 `src/infer/sdl_converter.py` 中的抖动参数：

```python
# Floyd-Steinberg 误差扩散系数（默认 0.6）
DITHER_COEFFICIENT = 0.6   # 降低 → 更保守的扩散，升高 → 更激进的扩散

# 严格验证模式
STRICT_VALIDATE_OUTPUT = True  # False → 允许少量超色域像素
```

---

## 目录结构

```
final_lab/
├── README.md                          # 本文件
├── data/                              # 训练数据
│   ├── SDL2_0.txt                     # 物理 LED 色域边界坐标
│   ├── light_effect_captions.jsonl    # 图像-标题对标注
│   └── images/                        # 2000 张光效图像 (768×768)
│
├── src/                               # 源代码
│   ├── data/                          # 数据生成与加载
│   │   ├── simulate_graph_one.py      # 双色线性渐变生成
│   │   ├── simulate_graph_more.py     # 多色流体光晕生成
│   │   ├── generate_light_caption.py  # VLM 图像标注
│   │   ├── generate_random_prompt.py  # 随机提示词采样
│   │   ├── build_dataset_pipeline.py  # 数据集构建编排器
│   │   └── dataset.py                 # PyTorch Dataset / DataLoader
│   │
│   ├── finetune/                      # 模型微调
│   │   ├── full_finetune.py           # 完整微调
│   │   └── lora_finetune.py           # LoRA 微调
│   │
│   ├── infer/                         # 推理流水线
│   │   ├── inference.py               # 核心推理引擎
│   │   ├── prompt_translator.py       # 模块1: LLM 场景→提示词
│   │   ├── sdl_converter.py           # 模块4: 物理色域约束
│   │   ├── pipeline_infer.py          # 完整推理流水线
│   │   └── web_app.py                 # Gradio Web 演示
│   │
│   ├── plot/                          # 可视化脚本
│   │   ├── plot_2_bar.py              # 消融实验柱状图
│   │   ├── plot_3_bar.py              # 三方圆对比柱状图
│   │   ├── visualize_dataset.py       # 数据集预览
│   │   └── visualize_frequency_transform.py  # 频域分析
│   │
│   ├── hf/                            # Hugging Face 上传
│   │   ├── upload_dataset_to_hf.py
│   │   └── upload_model_to_hf.py
│   │
│   ├── evaluation.py                  # 评估框架
│   ├── compute_trainset_metrics.py    # 训练集统计
│   └── experiment.py                  # 超参数消融实验
│
├── runs/                              # 训练检查点
│   ├── full/                          # 完整微调（仅 MSE）
│   ├── full_color/                    # + 色彩损失
│   ├── full_color_high_freq/          # + 色彩 + 高频
│   ├── full_color_high_freq_text/     # + 色彩 + 高频 + 文本编码器 ⭐最佳
│   └── lora/                          # LoRA 微调
│
├── result/                            # 推理与评估结果
│   ├── eval/                          # 各检查点评估结果
│   │   ├── prompts.json               # 评估用固定提示词
│   │   ├── pretrain/                  # 预训练基座模型
│   │   ├── lora/                      # LoRA 模型
│   │   ├── full/                      # 完整微调（基线）
│   │   ├── full_color/                # + 色彩损失
│   │   └── full_color_high_freq_text/ # ⭐ 最佳检查点
│   └── experiment/                    # 消融实验结果
│
├── report/                            # 学术论文（LaTeX）
├── lecture/                           # 课程演讲材料
└── imgs/                              # 示例图片
```

---

## 已训练模型

| 模型 | 微调方式 | 损失函数 | CLIP Score ↑ | KID ↓ | 状态 |
|------|----------|----------|-------------|-------|------|
| `full` | 完整微调 | MSE | 0.282 | 0.072 | 基线 |
| `full_color` | 完整微调 | MSE + Color | 0.285 | 0.078 | 消融 |
| `full_color_high_freq` | 完整微调 | MSE + Color + HighFreq | — | — | 消融 |
| **`full_color_high_freq_text`** | **完整微调 + 文本编码器** | **MSE + Color + HighFreq** | **0.289** | **0.082** | **⭐ 最佳** |
| `lora` | LoRA | MSE | 0.285 | **0.095** | 轻量 |

---

## 评估结果

16 个固定提示词 × 8 张/提示词 = 128 张图像对比：

| 指标 | 预训练 SD1.5 | LoRA | 完整（基线） | 完整+色彩 | **完整+色彩+高频+TE** |
|------|:-----------:|:----:|:----------:|:--------:|:-------------------:|
| CLIP Score ↑ | 0.277 | 0.285 | 0.282 | 0.285 | **0.289** |
| KID ↓ | 0.216 | 0.095 | 0.072 | 0.078 | 0.082 |
| Prompt Consistency ↑ | 0.766 | **0.920** | 0.917 | 0.915 | 0.910 |
| Scene Leakage ↓ | 0.231 | **0.220** | 0.225 | 0.225 | 0.226 |
| High-Freq Energy ↓ | 4.37e-3 | 9.7e-7 | 7.9e-6 | 1.3e-5 | 3.0e-5 |

**结论**：微调后 KID 下降 67%，提示一致性从 0.766 提升至 0.92。LoRA 在提示一致性和场景泄露方面最优；完整微调（含色彩+高频损失+文本编码器）在 CLIP 分数上最优。

---

## 核心创新点

1. **色彩感知损失（Color Loss）** — 在训练过程中将预测 x₀ 解码到 CIELab 空间，对低色度和低饱和度施加惩罚，解决扩散模型生成"灰蒙蒙"的固有问题

2. **频域高频正则化（High-Freq Loss）** — 对亮度信号做 rFFT 并使用十字形低频掩码，抑制高频噪声，促使模型生成更平滑的色彩渐变

3. **Lab 空间 Floyd-Steinberg 抖动** — 在 CIE Lab 感知均匀空间中执行蛇形扫描误差扩散，将超色域颜色映射到物理 LED 色域，消除简单最近邻映射的条带伪影

4. **语义泄露防控体系** — 30+ 禁止场景词汇过滤 + JSON 结构化输出约束 + 强制后缀 "no room, no furniture, no objects"，确保模型学到的是"光效"而非"场景"

---

## 引用

如果本项目对你的研究或工作有帮助，请注明出处。

```
@misc{lighting-agent-2024,
  title   = {基于大模型微调与物理色域约束的高保真光效智能体设计},
  author  = {SMF-an},
  note    = {CV Course Final Project},
  url     = {<your-repo-url>}
}
```
