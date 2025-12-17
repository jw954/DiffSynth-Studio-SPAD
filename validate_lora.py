"""
FLUX LoRA Validation - Run inference on validation set
"""
import argparse
import torch
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="FLUX LoRA validation on val set")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--metadata_csv", type=str, default="/home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv", help="Validation CSV")
    parser.add_argument("--output_dir", type=str, default="./validation_outputs", help="Output directory")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--embedded_guidance", type=float, default=3.5, help="Embedded guidance")
    parser.add_argument("--denoising_strength", type=float, default=1.0, help="Denoising strength")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (None=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input").mkdir(exist_ok=True)
    (output_dir / "output").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    
    print("="*60)
    print("FLUX LoRA Validation")
    print("="*60)
    print(f"Checkpoint: {args.lora_checkpoint}")
    print(f"Validation CSV: {args.metadata_csv}")
    print(f"Output: {output_dir}")
    print("="*60)
    print()
    
    # Load validation samples
    print(f"Loading validation metadata...")
    samples = []
    with open(args.metadata_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    
    print(f"Processing {len(samples)} validation samples")
    
    # Load pipeline
    print("Loading FLUX pipeline...")
    vram_config = {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    
    vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5
    
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors", **vram_config),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors", **vram_config),
            ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", **vram_config),
        ],
        vram_limit=vram_limit,
    )
    
    print(f"Loading LoRA: {args.lora_checkpoint}")
    pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=1.0)
    
    # Run inference
    print("Running inference on validation set...")
    csv_base = Path(args.metadata_csv).parent
    
    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        # Load SPAD input and ground truth
        input_path = csv_base / sample['input_image']
        gt_path = csv_base / sample['image']
        
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Generate
        with torch.no_grad():
            result = pipe(
                prompt=sample.get('prompt', ''),
                input_image=input_img,
                denoising_strength=args.denoising_strength,
                height=512,
                width=512,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                embedded_guidance=args.embedded_guidance,
                seed=args.seed + idx,  # Different seed per sample
                rand_device="cuda",
            )
        
        # Save
        input_img.save(output_dir / "input" / f"input_{idx:04d}.png")
        result.save(output_dir / "output" / f"output_{idx:04d}.png")
        gt_img.save(output_dir / "ground_truth" / f"gt_{idx:04d}.png")
    
    print()
    print("✅ Validation complete!")
    print(f"   Input: {output_dir / 'input'}")
    print(f"   Output: {output_dir / 'output'}")
    print(f"   Ground truth: {output_dir / 'ground_truth'}")


if __name__ == "__main__":
    main()


