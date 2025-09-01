---
title: "Morphix"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "4.44.1"
app_file: "app.py"
pinned: true
---

# 🎨 Morphix – Portrait Style Transformation with Stable Diffusion + ControlNet

**Morphix** is an AI app that lets you upload a **portrait photo** and transform it into different **artistic styles** (Comic, Anime, Oil Painting, Pixel Art, Watercolor).  
It uses **Stable Diffusion + ControlNet (Canny)** to preserve structure while changing the artistic look.

---

## ✨ Features
- Upload any portrait photo.
- Choose from 5 pre-defined artistic styles.
- Adjustable **guidance scale** and **inference steps** for creativity vs. accuracy.
- Runs interactively in your browser via **Gradio**.
- Ready to deploy on **Hugging Face Spaces**.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **PyTorch** with CUDA
- **Hugging Face Diffusers** (Stable Diffusion + ControlNet)
- **Gradio** for UI
- **OpenCV** for edge detection

---

## 📦 Installation

Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/Morphix.git
cd Morphix
pip install -r requirements.txt
```

Or using conda:
```bash
conda create -n morphix python=3.10 -y
conda activate morphix
pip install -r requirements.txt
```

### Run the Gradio app locally:
```bash
python app.py
```
