import gradio as gr
import torch
import os
import yaml
from PIL import Image, ImageOps
from diffusers.pipelines import FluxPipeline
import numpy as np
from datetime import datetime
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add relative path import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flux.condition import Condition
from flux.generate import generate

pipe = None
config = None


def load_config(config_path="ops/config/lineart_512.yaml"):
    """Load configuration file"""
    global config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_latest_checkpoint(runs_dir):
    """Load the latest checkpoint"""
    # Get the latest run directory
    run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    if not run_dirs:
        raise ValueError("No run directories found")
    latest_run = sorted(run_dirs)[-1]
    
    # Get checkpoint directory
    ckpt_dir = os.path.join(runs_dir, latest_run, "ckpt")
    if not os.path.exists(ckpt_dir):
        raise ValueError(f"No checkpoint directory found in {latest_run}")
    
    # Get the latest checkpoint file
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.safetensors')]
    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {ckpt_dir}")
    latest_ckpt = sorted(ckpt_files)[-1]
    
    return os.path.join(ckpt_dir, latest_ckpt)

def extract_lineart(img):
    # load model from lllyasviel/Annotators sk_model
    from annotator.lineart import BatchLineartDetector
    annotator_ckpts_path = "annotators"
    preprocessor = BatchLineartDetector(annotator_ckpts_path)
    preprocessor.to(device,dtype=torch.float32)
    # Convert to tensor and add batch dimension
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to [C, H, W] with value range [0, 1]
    ])
    input_tensor = transform(img).unsqueeze(0)  # Become [1, 3, H, W]
    input_tensor = input_tensor.to('cuda')
    
    # Process image and extract lineart
    lineart_img = preprocessor(input_tensor).squeeze().cpu().numpy()
    lineart_img = (lineart_img * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8 image format
    lineart_img_convert = Image.fromarray(255 - np.array(lineart_img))
    
    # Directly convert to RGB format without saving temporary files
    lineart_img_convert = lineart_img_convert.convert("RGB")
    return lineart_img_convert

def prepare_images(lineart_img, ref_img, condition_size=512):
    """Prepare input images"""
    # For now, use fixed square size to avoid shape mismatch issues
    # The model expects specific dimensions that are trained on
    target_width = condition_size
    target_height = condition_size
    
    # Resize images to square format
    lineart_img = lineart_img.resize((target_width, target_height)).convert("RGBA")
    # Convert to white background
    white_bg = Image.new("RGBA", lineart_img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white_bg, lineart_img)
    lineart_img = composited.convert("RGB")
    
    ref_img = ref_img.resize((target_width, target_height)).convert("RGBA")
    # Convert to black background
    bg = Image.new("RGBA", ref_img.size, (0, 0, 0, 255))
    composited = Image.alpha_composite(bg, ref_img)
    ref_img = composited.convert("RGB")
    
    return lineart_img, ref_img, target_width, target_height, 1.0


def init_pipeline(config, checkpoint_path):
    """Initialize model pipeline"""
    global pipe
    # Initialize FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        config["flux_path"], 
        torch_dtype=getattr(torch, config["dtype"])
    )
    pipe = pipe.to("cuda")
    
    # Load LoRA weights
    pipe.load_lora_weights(
        checkpoint_path,
        adapter_name="lineart_ref"
    )
    
    return pipe


def extract_lineart_from_image(lineart_image):
    """Extract lineart from uploaded image and return it"""
    if lineart_image is None:
        return None, "Please upload an image first"
    
    try:
        # Extract lineart using the extract_lineart function
        extracted_lineart = extract_lineart(lineart_image)
        return extracted_lineart, "Lineart extracted successfully!"
    except Exception as e:
        return None, f"Failed to extract lineart: {str(e)}"


def process_images(reference_image, lineart_image, seed=42):
    """Process reference image and lineart image to generate colorized image"""
    if reference_image is None or lineart_image is None:
        return None, "Please upload reference image and lineart image"
    
    try:
        global pipe, config
        
        # Load configuration
        if config is None:
            config = load_config()
        
        # Set paths
        runs_dir = "runs/20250709-084341"
        checkpoint_path = os.path.join(runs_dir, "ckpt", "50000", "pytorch_lora_weights.safetensors")
        
        # Initialize model
        if pipe is None:
            pipe = init_pipeline(config, checkpoint_path)
        
        # Prepare input images
        condition_size = config["train"]["dataset"]["condition_size"]
        lineart_img, ref_img, target_width, target_height, aspect_ratio = prepare_images(lineart_image, reference_image, condition_size)

        # Create conditions
        condition_1 = Condition(
            condition_type="lineart",
            condition=lineart_img,
            position_delta=[0, 0]
        )
        condition_2 = Condition(
            condition_type="reference",
            condition=ref_img,
            position_delta=[0, 0]
        )
        
        # Generate image with standard target size from config
        target_size = config["train"]["dataset"]["target_size"]
        with torch.no_grad():
            result = generate(
                pipe,
                prompt="",
                conditions=[condition_1, condition_2],
                height=target_size,
                width=target_size,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                model_config=config.get("model", {}),
                default_lora=True
            )
        
        # Get the generated image
        generated_image = result.images[0]
        
        # Calculate original aspect ratio for display
        original_width, original_height = lineart_image.size
        original_aspect_ratio = original_width / original_height
        
        # Resize the generated image to match original aspect ratio for display
        if original_aspect_ratio >= 1:  # Landscape or square
            display_width = target_size
            display_height = int(target_size / original_aspect_ratio)
        else:  # Portrait
            display_height = target_size
            display_width = int(target_size * original_aspect_ratio)
        
        # Ensure display dimensions are reasonable
        display_width = max(display_width, 256)
        display_height = max(display_height, 256)
        
        # Resize for display
        generated_image = generated_image.resize((display_width, display_height))
        
        return generated_image, "Generation successful!"
        
    except Exception as e:
        return None, f"Generation failed: {str(e)}"


def get_samples():
    """Get sample images"""
    # Manually specify reference and lineart image pairs
    examples = [
        ["samples/bg_6_1.png", "samples/bg_6_2_line.png"],
        ["samples/bg_7_1.png", "samples/bg_7_2_line.png"],
        ["samples/30_1.png", "samples/30_2_line.png"],
        ["samples/1436.png", "samples/1604_line.png"],
        ["samples/15225.png", "samples/15281_line.png"],
    ]
    return examples


def create_demo():
    # Create Gradio interface
    with gr.Blocks(title="MangaDiT - Lineart Coloring Tool") as demo:
        gr.Markdown("# MangaDiT - Reference-based Lineart Coloring Tool")
        gr.Markdown("Upload a reference image and a lineart image, and then generate a colored image.")
        
        with gr.Row():
            with gr.Column():
                reference_image = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=300
                )
                lineart_image = gr.Image(
                    label="Lineart Image", 
                    type="pil",
                    height=300
                )
                gr.Markdown("**Extract Lineart**: Click to extract lineart from the uploaded image if it's not already a lineart or the lineart is not good")
                extract_lineart_btn = gr.Button(
                    "Extract Lineart",
                    variant="secondary"
                )
                seed_input = gr.Slider(
                    minimum=0,
                    maximum=999999,
                    value=42,
                    step=1,
                    label="Random Seed"
                )
                generate_btn = gr.Button("Generate Colored Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="Generated Colored Image",
                    type="pil",
                    height=400
                )
                status_text = gr.Textbox(
                    label="Status",
                    value="Waiting for input...",
                    interactive=False
                )
        
        # Add examples
        gr.Examples(
            examples=get_samples(),
            inputs=[reference_image, lineart_image],
            label="Sample Images"
        )
        
        # Bind events
        extract_lineart_btn.click(
            fn=extract_lineart_from_image,
            inputs=[lineart_image],
            outputs=[lineart_image, status_text]
        )
        
        generate_btn.click(
            fn=process_images,
            inputs=[reference_image, lineart_image, seed_input],
            outputs=[output_image, status_text]
        )
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MangaDiT")
    parser.add_argument("--share", action="store_true", help="Create a public link to demo")
    parser.add_argument("--port", type=int, default=7888)
    args = parser.parse_args()
    
    demo = create_demo()
    
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
