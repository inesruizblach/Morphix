import gradio as gr
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image

# Device detection and dtype setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype
)

# Load Stable Diffusion with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Artistic style prompts
STYLE_PROMPTS = {
    "Comic": "portrait of a person, comic book style, bold black outlines, vibrant colors, pop art aesthetic",
    "Anime": "portrait of a person, anime style, Studio Ghibli aesthetic, detailed, cinematic lighting",
    "Oil Painting": "portrait of a person, oil painting, Rembrandt style, dramatic shadows, baroque aesthetic",
    "Pixel Art": "portrait of a person, retro pixel art, 16-bit game style, sharp pixels, vibrant colors",
    "Watercolor": "portrait of a person, watercolor style, soft pastel colors, brush stroke texture"
}

def preprocess_image(image):
    """
    Preprocess the uploaded image for the pipeline.
    - Converts image to RGB.
    - Resizes to 512x512 pixels to avoid batch size issues.
    - Ensures a consistent input format for Stable Diffusion + ControlNet.

    Args:
        image (PIL.Image): Input portrait uploaded by the user.

    Returns:
        PIL.Image: Preprocessed image ready for edge detection.
    """
    return image.convert("RGB").resize((512, 512))

def canny_edge(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection to a preprocessed image.

    Args:
        image (PIL.Image): Preprocessed image.
        low_threshold (int): Lower threshold for Canny detection.
        high_threshold (int): Upper threshold for Canny detection.

    Returns:
        PIL.Image: Edge-detected image with 3 channels for ControlNet input.
    """
    image = np.array(image)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)

def generate(image, style, guidance_scale=7.5, steps=30):
    """
    Generate a stylized portrait using Stable Diffusion + ControlNet.

    Args:
        image (PIL.Image): Input portrait.
        style (str): Artistic style key from STYLE_PROMPTS.
        guidance_scale (float): Prompt adherence strength.
        steps (int): Number of inference steps.

    Returns:
        PIL.Image: AI-generated stylized portrait.
    """
    image = preprocess_image(image)
    edges = canny_edge(image)
    prompt = STYLE_PROMPTS[style]
    result = pipe(prompt, image=edges, num_inference_steps=steps, guidance_scale=guidance_scale)
    return result.images[0]

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(
        "# 🎨 Morphix – Transform Portraits into Artistic Styles\n"
        "**Tip:** Please upload a portrait with the face centered for best results."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload a centered portrait")
            style = gr.Dropdown(list(STYLE_PROMPTS.keys()), value="Comic", label="Choose style")
            guidance = gr.Slider(5, 12, value=7.5, step=0.5, label="Guidance Scale")
            steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
            btn = gr.Button("Generate")

        with gr.Column():
            output = gr.Image(label="Stylized Portrait")

    btn.click(fn=generate, inputs=[input_image, style, guidance, steps], outputs=output)

if __name__ == "__main__":
    demo.launch()
