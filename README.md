# 📄 Document-to-Markdown with QLoRA Fine-Tuned Qwen2-VL

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20x2-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Fine-tuning Qwen2-VL-2B-Instruct with QLoRA to convert document images into structured Markdown**

*AI4009 Generative AI — Assignment 5 | FAST-NUCES Spring 2026*

[📓 Kaggle Notebook](#) · [📝 Medium Blog](#) · [🤗 Model](#)

</div>

---

## 📌 Overview

This project fine-tunes the **Qwen2-VL-2B-Instruct** Vision Language Model (VLM) using **QLoRA** (Quantized Low-Rank Adaptation) to perform document understanding — specifically, converting a scanned or photographed document page into clean, structured Markdown text.

```
Input  →  [Document Image]
Output →  # Heading
           - bullet list
           | table | cols |
           $$equation$$
```

---

## ✨ Features

- ✅ End-to-end VLM fine-tuning pipeline (image → Markdown)
- ✅ 4-bit NF4 quantization via BitsAndBytes
- ✅ LoRA adapters (rank 16) on attention + MLP layers
- ✅ ChatML format dataset preparation
- ✅ ROUGE-1/2/L and BLEU evaluation
- ✅ Zero-shot vs fine-tuned qualitative comparison
- ✅ Gradio web app for live inference
- ✅ Runs on free Kaggle dual T4 GPUs

---

## 🗂️ Repository Structure

```
📦 vlm-qlora-doc2markdown/
├── 📓 AI_ASS05_22F_8816.ipynb     # Main Kaggle notebook (full pipeline)
├── 📁 outputs/
│   ├── loss_curves.png             # Training & validation loss plot
│   ├── val_comparison_*.png        # Ground truth vs generated comparisons
│   ├── zeroshot_vs_finetuned.png   # Zero-shot vs fine-tuned comparison
│   └── unseen_result_*.png         # Inference on unseen documents
├── 📁 checkpoints/
│   ├── best_checkpoint/            # Best model by validation loss
│   └── final_checkpoint/           # Final epoch checkpoint
└── 📄 README.md
```

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Vision encoder | Processes image → visual token sequence |
| Language decoder | Generates Markdown token by token |
| Fine-tuning method | QLoRA (4-bit NF4 + LoRA rank 16) |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable params | ~1–2% of total model parameters |

---

## 📊 Dataset

**Nougat Training Dataset Example**
- Source: [Kaggle](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example)
- Format: Document page images paired with `.mmd` (Markdown) ground truth files
- Content: Academic papers, technical documents, forms — covering headings, tables, equations, lists, and code blocks
- Split: **80% train / 20% validation**

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 1 |
| Gradient accumulation | 8 steps (effective batch = 8) |
| Learning rate | 1.5e-4 |
| LR schedule | Cosine with 5% warmup |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Image resolution | 640px (longest edge) |
| Precision | bfloat16 (mixed precision) |
| Quantization | 4-bit NF4 (double quant enabled) |
| Platform | Kaggle GPU T4 × 2 |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/vlm-qlora-doc2markdown.git
cd vlm-qlora-doc2markdown
```

### 2. Install dependencies

```bash
pip install transformers==4.45.0 peft==0.12.0 bitsandbytes==0.44.0 \
    accelerate==0.34.0 datasets trl einops qwen-vl-utils \
    torchvision pillow rouge-score nltk gradio
```

### 3. Run the notebook

Open `AI_ASS05_22F_8816.ipynb` on Kaggle with GPU T4 x2 enabled, or locally if you have sufficient VRAM.

### 4. Launch the Gradio app

The final cell of the notebook launches a Gradio interface automatically:

```python
demo.launch(share=True)
```

Upload any document image and get structured Markdown output instantly.

---

## 📈 Results

### Loss Curves

Training and validation loss decreased consistently across all 3 epochs, confirming the model learned the task without overfitting.

### Evaluation Metrics (Validation Set)

| Metric | Score |
|---|---|
| ROUGE-1 | see notebook |
| ROUGE-2 | see notebook |
| ROUGE-L | see notebook |
| BLEU | see notebook |

### Zero-Shot vs Fine-Tuned

| | Base Model (Zero-Shot) | QLoRA Fine-Tuned |
|---|---|---|
| Markdown syntax | Often ignored | Consistently correct |
| Table structure | Missed | Reproduced accurately |
| Headings | Generic | Correctly levelled |
| ROUGE-L | lower | significantly higher |

---

## 🖥️ Gradio App

The deployed app accepts any document image and returns:
- Raw Markdown text output
- Live rendered Markdown preview

```
Upload image → Click Generate → Get Markdown
```

---

## 📋 Assignment Tasks Completed

- [x] Part 1 — Dataset exploration & statistics
- [x] Part 2 — Data preparation in ChatML format
- [x] Part 3 — 80/20 train/validation split
- [x] Part 4 — QLoRA fine-tuning (4-bit + LoRA rank 16)
- [x] Part 5 — Markdown generation with ROUGE/BLEU evaluation
- [x] Part 6 — Testing on 3 training + 3 unseen document images
- [x] Visualization — Side-by-side ground truth vs generated comparison
- [x] App — Gradio deployment
- [x] Bonus — Zero-shot vs fine-tuned comparison

---

## 🔗 Links

| Resource | Link |
|---|---|
| Kaggle Notebook | [Open on Kaggle](#) |
| Medium Blog Post | [Read on Medium](#) |
| LinkedIn Post | [View on LinkedIn](#) |
| Base Model (HF) | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| Dataset | [Nougat Dataset on Kaggle](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example) |

---

## 👤 Author

**Naeem** — Roll No. 22F-8816
FAST-NUCES | AI4009 Generative AI | Spring 2026

---

## 📄 License

This project is for academic purposes under the MIT License.
