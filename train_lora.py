"""
FLUX ControlNet LoRA Training with TensorBoard + Image Logging
Based on examples/flux/model_training/train.py with added logging features
"""
import torch, os, argparse, accelerate, re
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.diffusion import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FluxTrainingModule(DiffusionTrainingModule):
    """Official FLUX training module - unchanged from examples/flux/model_training/train.py"""
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_1_path=None, tokenizer_2_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_1_config = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer/") if tokenizer_1_path is None else ModelConfig(tokenizer_1_path)
        tokenizer_2_config = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer_2/") if tokenizer_2_path is None else ModelConfig(tokenizer_2_path)
        self.pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_1_config=tokenizer_1_config, tokenizer_2_config=tokenizer_2_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # input_image is the GROUND TRUTH target for training
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Training parameters
            "cfg_scale": 1,
            "embedded_guidance": 1,
            "t5_sequence_length": 512,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        # Parse extra inputs (controlnet_image, etc.)
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def flux_parser():
    """Official FLUX argument parser"""
    parser = argparse.ArgumentParser(description="FLUX ControlNet LoRA training with logging")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_1_path", type=str, default=None, help="Path to CLIP tokenizer.")
    parser.add_argument("--tokenizer_2_path", type=str, default=None, help="Path to T5 tokenizer.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    # Custom logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs/flux_spad_lora", help="TensorBoard log directory")
    parser.add_argument("--log_freq", type=int, default=300, help="Scalar logging frequency (steps)")
    parser.add_argument("--image_log_freq", type=int, default=1000, help="Image logging frequency (steps)")
    return parser


def convert_lora_format(state_dict, alpha=None):
    """Official LoRA format converter"""
    prefix_rename_dict = {
        "single_blocks": "lora_unet_single_blocks",
        "blocks": "lora_unet_double_blocks",
    }
    middle_rename_dict = {
        "norm.linear": "modulation_lin",
        "to_qkv_mlp": "linear1",
        "proj_out": "linear2",
        "norm1_a.linear": "img_mod_lin",
        "norm1_b.linear": "txt_mod_lin",
        "attn.a_to_qkv": "img_attn_qkv",
        "attn.b_to_qkv": "txt_attn_qkv",
        "attn.a_to_out": "img_attn_proj",
        "attn.b_to_out": "txt_attn_proj",
        "ff_a.0": "img_mlp_0",
        "ff_a.2": "img_mlp_2",
        "ff_b.0": "txt_mlp_0",
        "ff_b.2": "txt_mlp_2",
    }
    suffix_rename_dict = {
        "lora_B.weight": "lora_up.weight",
        "lora_A.weight": "lora_down.weight",
    }
    state_dict_ = {}
    for name, param in state_dict.items():
        names = name.split(".")
        if names[-2] != "lora_A" and names[-2] != "lora_B":
            names.pop(-2)
        prefix = names[0]
        middle = ".".join(names[2:-2])
        suffix = ".".join(names[-2:])
        block_id = names[1]
        if middle not in middle_rename_dict:
            continue
        rename = prefix_rename_dict[prefix] + "_" + block_id + "_" + middle_rename_dict[middle] + "." + suffix_rename_dict[suffix]
        state_dict_[rename] = param
        if rename.endswith("lora_up.weight"):
            lora_alpha = alpha if alpha is not None else param.shape[-1]
            state_dict_[rename.replace("lora_up.weight", "alpha")] = torch.tensor((lora_alpha,))[0]
    return state_dict_


def log_sample_images(model, data, tb_writer, global_step, device="cuda"):
    """
    Generate and log sample images to TensorBoard during training.
    Shows: SPAD input (controlnet_image), VAE reconstruction, generated sample, ground truth
    """
    import numpy as np
    from PIL import Image as PILImage
    from diffsynth.utils.controlnet import ControlNetInput
    
    model.eval()
    with torch.no_grad():
        try:
            # Extract data
            controlnet_img_pil = data.get('controlnet_image')  # SPAD conditioning
            gt_img_pil = data['image']  # Ground truth RGB
            prompt = data.get('prompt', '')
            
            if controlnet_img_pil is None:
                print("Warning: No controlnet_image in batch, skipping image logging")
                model.train()
                return
            
            pipe = model.pipe
            
            # 1. Convert SPAD input to tensor for logging
            input_tensor = pipe.preprocess_image(controlnet_img_pil)  # [-1, 1]
            
            # 2. Get VAE reconstruction of ground truth (for reference)
            gt_tensor = pipe.preprocess_image(gt_img_pil)
            with torch.no_grad():
                gt_latent = pipe.vae_encoder(gt_tensor)
                vae_recon = pipe.vae_decoder(gt_latent)
            
            # 3. Generate sample using current LoRA + ControlNet
            print(f"[Step {global_step}] Generating sample with ControlNet...")
            
            # Create ControlNetInput wrapper for inference
            controlnet_inputs = [ControlNetInput(image=controlnet_img_pil)]
            
            generated = pipe(
                prompt=prompt,
                input_image=gt_img_pil,  # Ground truth for denoising_strength
                controlnet_inputs=controlnet_inputs,  # SPAD conditioning
                denoising_strength=1.0,  # Full generation from noise
                height=448,  # Match training size (reduced for VRAM)
                width=448,
                num_inference_steps=10,  # Fast sampling
                cfg_scale=1.0,
                embedded_guidance=3.5,
                seed=42,
                rand_device=device,
            )
            
            # CRITICAL: Restore scheduler to training mode
            pipe.scheduler.set_timesteps(1000, training=True)
            
            # Convert generated PIL to tensor
            gen_np = np.array(generated)
            gen_tensor = torch.from_numpy(gen_np).permute(2, 0, 1).float() / 255.0
            gen_tensor = gen_tensor.unsqueeze(0).to(device) * 2.0 - 1.0  # [-1, 1]
            
            # Normalize for display [0, 1]
            def normalize_for_display(img):
                return (img + 1.0) / 2.0
            
            # Log to TensorBoard
            tb_writer.add_image("samples/1_spad_input", normalize_for_display(input_tensor[0]), global_step)
            tb_writer.add_image("samples/2_vae_reconstruction", normalize_for_display(vae_recon[0]), global_step)
            tb_writer.add_image("samples/3_generated_sample", normalize_for_display(gen_tensor[0]), global_step)
            tb_writer.add_image("samples/4_ground_truth", normalize_for_display(gt_tensor[0]), global_step)
            
            print(f"✓ Logged: SPAD → VAE → GENERATED → GT")
            
        except Exception as e:
            import traceback
            print(f"Warning: Image logging failed: {e}")
            traceback.print_exc()
    
    model.train()


def parse_resume_epoch(checkpoint_path):
    """Parse epoch number from checkpoint path like 'epoch-4.safetensors' -> 4"""
    if checkpoint_path is None:
        return None
    match = re.search(r'epoch[-_]?(\d+)', checkpoint_path, re.IGNORECASE)
    return int(match.group(1)) if match else None


def launch_training_with_logging(
    accelerator: accelerate.Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    tb_writer: SummaryWriter,
    log_freq: int,
    image_log_freq: int,
    resume_epoch: int,
    args,
):
    """
    Enhanced training loop with TensorBoard logging.
    Based on diffsynth.diffusion.runner.launch_training_task
    """
    # Setup
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    
    start_epoch = 0 if resume_epoch is None else resume_epoch + 1
    if start_epoch > 0:
        print(f"\n{'='*60}")
        print(f"🔄 RESUMING from epoch {resume_epoch}")
        print(f"   Starting at epoch {start_epoch}, training to {num_epochs - 1}")
        print(f"{'='*60}\n")
    
    # Optimizer and dataloader
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Calculate global_step for resume
    steps_per_epoch = len(dataloader)
    global_step = start_epoch * steps_per_epoch
    
    # Training loop
    for epoch_id in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_id+1}/{num_epochs}")
        for data in progress_bar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # Forward pass
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                
                # Logging
                loss_value = loss.item()
                epoch_loss += loss_value
                epoch_steps += 1
                global_step += 1
                
                # TensorBoard scalars
                if global_step % log_freq == 0 or global_step == 1:
                    tb_writer.add_scalar("train/loss", loss_value, global_step)
                    tb_writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)
                    tb_writer.add_scalar("train/epoch", epoch_id, global_step)
                
                # TensorBoard images
                if global_step % image_log_freq == 0 and accelerator.is_main_process:
                    try:
                        log_sample_images(model, data, tb_writer, global_step, accelerator.device)
                    except Exception as e:
                        print(f"Warning: Image logging failed: {e}")
                
                # Update progress
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'step': global_step
                })
                
                # Checkpoint saving
                model_logger.on_step_end(accelerator, model, save_steps)
        
        # Epoch end
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"\n[Epoch {epoch_id+1}] Average loss: {avg_loss:.4f}")
        tb_writer.add_scalar("train/epoch_loss", avg_loss, epoch_id)
        
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    
    # Training end
    model_logger.on_training_end(accelerator, model, save_steps)
    print(f"\n✅ Training complete! Total steps: {global_step}")


if __name__ == "__main__":
    # Parse arguments
    parser = flux_parser()
    args = parser.parse_args()
    
    # Setup accelerator
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    
    # Dataset
    print(f"[Dataset] Loading from: {args.dataset_metadata_path}")
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    print(f"[Dataset] {len(dataset)} samples (repeat={args.dataset_repeat})")
    
    # Resume epoch
    resume_epoch = parse_resume_epoch(args.lora_checkpoint)
    if resume_epoch is not None:
        print(f"[Resume] Detected checkpoint from epoch {resume_epoch}")
    
    # Model
    print(f"[Model] Initializing FLUX ControlNet LoRA...")
    model = FluxTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_1_path=args.tokenizer_1_path,
        tokenizer_2_path=args.tokenizer_2_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
    )
    
    # Model logger (checkpoints)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        state_dict_converter=convert_lora_format if args.align_to_opensource_format else lambda x:x,
    )
    
    # TensorBoard
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[TensorBoard] Logging to: {log_dir}")
    
    # Launch training
    print(f"\n{'='*60}")
    print(f"Starting FLUX ControlNet LoRA Training")
    print(f"{'='*60}")
    print(f"  Epochs: {start_epoch if resume_epoch else 0} → {args.num_epochs - 1}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Checkpoints: {args.output_path}")
    print(f"{'='*60}\n")
    
    launch_training_with_logging(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        tb_writer=tb_writer,
        log_freq=args.log_freq,
        image_log_freq=args.image_log_freq,
        resume_epoch=resume_epoch,
        args=args,
    )
    
    tb_writer.close()
    print(f"\n✅ Complete! Checkpoints: {args.output_path} | Logs: {log_dir}")
