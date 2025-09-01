import gradio as gr
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Load ControlNet pre-trained for Canny edge detection
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

# Load Stable Diffusion with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")

# Artistic style prompts
STYLE_PROMPTS = {
    "Comic": "portrait of a person, comic book style, bold black outlines, vibrant colors, pop art aesthetic",
    "Anime": "portrait of a person, anime style, Studio Ghibli aesthetic, detailed, cinematic lighting",
    "Oil Painting": "portrait of a person, oil painting, Rembrandt style, dramatic shadows, baroque aesthetic",
    "Pixel Art": "portrait of a person, retro pixel art, 16-bit game style, sharp pixels, vibrant colors",
    "Watercolor": "portrait of a person, watercolor style, soft pastel colors, brush stroke texture"
}

def canny_edge(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection to the input image.

    Args:
        image (PIL.Image): Input portrait.
        low_threshold (int): Lower threshold for edges.
        high_threshold (int): Upper threshold for edges.

    Returns:
        np.ndarray: 3-channel edge-detected image for ControlNet.
    """
    image = np.array(image)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return edges

def generate(image, style, guidance_scale=7.5, steps=30):
    """
    Generate a stylized portrait based on the selected style.

    Args:
        image (PIL.Image): Input portrait.
        style (str): Key from STYLE_PROMPTS.
        guidance_scale (float): Prompt adherence strength.
        steps (int): Number of inference steps.

    Returns:
        PIL.Image: Generated stylized portrait.
    """
    edges = canny_edge(image)
    prompt = STYLE_PROMPTS[style]

    result = pipe(
        prompt,
        image=edges,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )
    return result.images[0]

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 StyleMorph – Transform Portraits into Artistic Styles")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload a portrait")
            style = gr.Dropdown(list(STYLE_PROMPTS.keys()), value="Comic", label="Choose style")
            guidance = gr.Slider(5, 12, value=7.5, step=0.5, label="Guidance Scale")
            steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
            btn = gr.Button("Generate")

        with gr.Column():
            output = gr.Image(label="Stylized Portrait")

    btn.click(fn=generate, inputs=[input_image, style, guidance, steps], outputs=output)

if __name__ == "__main__":
    demo.launch()
