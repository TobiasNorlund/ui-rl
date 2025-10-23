"""VLM (Vision-Language Model) wrapper for UI interaction using HuggingFace models."""

from typing import Dict, Any, Optional, List, Union
import logging
import threading
import torch
import torch.nn as nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VLMWrapper(nn.Module):
    """
    Wrapper for HuggingFace Vision-Language Models for UI interaction.

    Supports models like:
    - Qwen2-VL
    - LLaVA
    - Idefics2
    - Florence-2
    - PaliGemma
    - Any HuggingFace VLM that follows ImageTextToText architecture

    Design:
    - Inherits from nn.Module for PyTorch compatibility
    - Handles image preprocessing via HuggingFace processor
    - Supports action generation from model outputs
    - Can be fine-tuned end-to-end or with LoRA
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
        lora_target_option: str = "attention+mlp",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Initialize VLM wrapper with HuggingFace model.

        Args:
            model_name: HuggingFace model identifier (default: "Qwen/Qwen2.5-VL-3B-Instruct")
            device: Device to load model on
            torch_dtype: Data type for model weights (float16 for efficiency)
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_config: LoRA configuration dict (rank, alpha, dropout, etc.)
            lora_target_option: Which layers to target with LoRA:
                - "attention": Only attention layers (q, k, v, o projections)
                - "mlp": Only MLP/FFN layers (gate, up, down projections)
                - "attention+mlp": Both attention and MLP (recommended)
                - "all-linear": All linear layers automatically
                - "custom": Use custom_modules from lora_config
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (vs greedy decoding)
        """
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_target_option = lora_target_option
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self._execution_lock = threading.RLock()

        logger.info(f"Loading VLM: {model_name}")

        # Load processor (handles image + text preprocessing)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor, trying tokenizer: {e}")
            self.processor = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
            )
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config)

        self.model.to(device)

        logger.info(f"VLM loaded successfully on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.2f}M")

    def _get_lora_target_modules(
        self,
        option: str,
        custom_modules: Optional[List[str]] = None
    ) -> Union[str, List[str]]:
        """
        Get target modules for LoRA based on targeting strategy.

        Args:
            option: One of "attention", "mlp", "attention+mlp", "all-linear", "custom"
            custom_modules: Custom module list (only used if option="custom")

        Returns:
            Target modules for LoRA config
        """
        if option == "attention":
            # Only attention layers
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

        elif option == "mlp":
            # Only MLP/FFN layers
            return ["gate_proj", "up_proj", "down_proj"]

        elif option == "attention+mlp":
            # Both attention and MLP (recommended per research)
            return [
                # Attention
                "q_proj", "k_proj", "v_proj", "o_proj",
                # MLP
                "gate_proj", "up_proj", "down_proj"
            ]

        elif option == "all-linear":
            # All linear layers (automatic detection by PEFT)
            return "all-linear"

        elif option == "custom":
            # User-specified modules
            if custom_modules is None:
                raise ValueError("custom_modules must be provided when option='custom'")
            return custom_modules

        else:
            raise ValueError(
                f"Unknown lora_target_option: {option}. "
                f"Must be one of: attention, mlp, attention+mlp, all-linear, custom"
            )

    def _print_trainable_parameters(self):
        """Print trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        percentage = 100 * trainable / total

        logger.info(f"Trainable params: {trainable:,} ({percentage:.2f}%)")
        logger.info(f"Total params: {total:,}")
        logger.info(f"LoRA params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    def _apply_lora(self, lora_config: Optional[Dict[str, Any]] = None):
        """Apply LoRA for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                lora_config = {}

            # Set defaults with option to override
            default_config = {
                "r": 32,  # Increased from 16 based on research
                "lora_alpha": 32,
                "target_modules": self._get_lora_target_modules(
                    self.lora_target_option,
                    lora_config.get("custom_modules")
                ),
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }

            # Merge user config with defaults (user config takes precedence)
            default_config.update(lora_config)

            logger.info(f"LoRA target option: {self.lora_target_option}")
            logger.info(f"LoRA target modules: {default_config['target_modules']}")
            logger.info(f"LoRA rank (r): {default_config['r']}")
            logger.info(f"LoRA alpha: {default_config['lora_alpha']}")

            config = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, config)

            logger.info("LoRA applied successfully")
            self._print_trainable_parameters()

        except ImportError:
            logger.error("peft library not installed. Install with: pip install peft")
            raise

    def forward(
        self,
        images: Union[torch.Tensor, List[Image.Image], np.ndarray],
        prompts: Union[str, List[str]],
        return_loss: bool = True,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            images: Input images [B, C, H, W] or list of PIL Images or numpy arrays
            prompts: Text prompts (single string or batch)
            return_loss: Whether to compute loss (for training)
            labels: Optional ground truth labels for supervised learning (token IDs)

        Returns:
            Dictionary containing:
            - logits: Model output logits
            - loss: Loss value (if return_loss=True and labels provided)
            - hidden_states: Optional hidden states
        """
        with self._execution_lock:
            processed_images: Union[List[Image.Image], torch.Tensor, np.ndarray, Image.Image]
            processed_images = images

            if isinstance(processed_images, torch.Tensor):
                processed_images = self._tensor_to_pil_images(processed_images)
            elif isinstance(processed_images, np.ndarray):
                processed_images = [Image.fromarray(processed_images.astype(np.uint8))]
            elif isinstance(processed_images, Image.Image):
                processed_images = [processed_images]

            processed_prompts = prompts
            if isinstance(processed_prompts, str):
                processed_prompts = [processed_prompts]

            inputs = self.processor(
                text=processed_prompts,
                images=processed_images,
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            if return_loss and labels is not None:
                inputs["labels"] = labels.to(self.device)

            outputs = self.model(**inputs)

            return {
                "logits": outputs.logits if hasattr(outputs, "logits") else None,
                "loss": outputs.loss if hasattr(outputs, "loss") else None,
                "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None
            }

    def predict_action(
        self,
        images: Union[torch.Tensor, Image.Image, np.ndarray],
        prompt: str
    ) -> str:
        """
        Generate action from image and prompt (inference mode).

        Args:
            images: Input image (single image)
            prompt: Task prompt

        Returns:
            Generated action text
        """
        with self._execution_lock:
            previous_mode = self.training
            self.eval()

            try:
                with torch.no_grad():
                    processed_image = images
                    if isinstance(processed_image, torch.Tensor):
                        if processed_image.dim() == 4:
                            processed_image = processed_image[0]
                        processed_image = self._tensor_to_pil_image(processed_image)
                    elif isinstance(processed_image, np.ndarray):
                        processed_image = Image.fromarray(processed_image.astype(np.uint8))

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": processed_image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    chat_prompt = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    inputs = self.processor(
                        text=chat_prompt,
                        images=processed_image,
                        return_tensors="pt",
                        padding=True
                    )

                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                    generated_text = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )[0]

                    if prompt in generated_text:
                        generated_text = generated_text.replace(prompt, "").strip()

                return generated_text
            finally:
                self.train(previous_mode)

    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert batch of tensors to list of PIL images."""
        images = []

        for i in range(tensor.shape[0]):
            img = tensor[i]
            img = self._tensor_to_pil_image(img)
            images.append(img)

        return images

    def _tensor_to_pil_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert single tensor to PIL image."""
        # Tensor is [C, H, W] or [H, W, C]
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3]:  # [C, H, W]
                tensor = tensor.permute(1, 2, 0)  # -> [H, W, C]

        # Denormalize if needed (assuming 0-1 range)
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).byte()

        # Convert to numpy
        img_np = tensor.cpu().numpy().astype(np.uint8)

        # Handle grayscale
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        return Image.fromarray(img_np)

    def save_pretrained(self, save_path: str):
        """Save model and processor."""
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        logger.info("Model saved successfully")

    def load_pretrained(self, load_path: str):
        """Load model and processor from path."""
        logger.info(f"Loading model from {load_path}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            load_path,
            dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            load_path,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters (only train language model)."""
        logger.info("Freezing vision encoder")
        for name, param in self.model.named_parameters():
            if "vision" in name.lower() or "visual" in name.lower():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after freezing: {trainable / 1e6:.2f}M")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        logger.info("Unfreezing all parameters")
        for param in self.model.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters after unfreezing: {trainable / 1e6:.2f}M")

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def execution_lock(self) -> threading.RLock:
        """Lock guarding model forward/generate calls for thread safety."""
        return self._execution_lock

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.model.parameters())
