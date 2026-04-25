import contextlib

import torch
from torch import nn

from .connectors import QFormerConnector

try:
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except ImportError:  # pragma: no cover - optional dependency
    AutoImageProcessor = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def require_transformers():
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for the server-stage scripts. "
            "Please install it in the server environment before running Stage 2-server or Stage 3-server."
        )


def require_peft():
    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is required for LoRA/QLoRA fine-tuning. "
            "Please install peft in the server environment before running Stage 3-server."
        )


def resolve_torch_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def build_server_tokenizer(model_name):
    require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def build_image_processor(model_name):
    require_transformers()
    if model_name is None:
        return None
    return AutoImageProcessor.from_pretrained(model_name)


class ServerQFormerLLM(nn.Module):
    def __init__(
        self,
        llm_model_name,
        vision_model_name=None,
        qformer_token_grid=(4, 4),
        num_query_tokens=32,
        qformer_layers=4,
        qformer_heads=8,
        qformer_dropout=0.1,
        freeze_llm=True,
        freeze_vision=True,
        use_lora=False,
        use_qlora=False,
        torch_dtype="float16",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
    ):
        super().__init__()
        require_transformers()

        dtype = resolve_torch_dtype(torch_dtype)
        llm_kwargs = {}
        if use_qlora:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes + transformers are required for QLoRA. "
                    "Please install bitsandbytes in the server environment."
                )
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            llm_kwargs["torch_dtype"] = dtype

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            **llm_kwargs,
        )
        self.vision_encoder = None
        if vision_model_name is not None:
            self.vision_encoder = AutoModel.from_pretrained(
                vision_model_name,
                torch_dtype=dtype,
            )
            if freeze_vision:
                freeze_module(self.vision_encoder)

        if freeze_llm and not use_lora:
            freeze_module(self.llm)

        if use_lora:
            require_peft()
            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.llm = get_peft_model(self.llm, lora_config)

        llm_hidden_size = self.llm.get_input_embeddings().embedding_dim
        self.connector = QFormerConnector(
            token_grid=qformer_token_grid,
            hidden_size=llm_hidden_size,
            num_query_tokens=num_query_tokens,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
            dropout=qformer_dropout,
            use_visual_tokens=vision_model_name is not None,
        )

    def encode_images(self, pixel_values):
        if self.vision_encoder is None or pixel_values is None:
            return None
        batch_size, num_images, channels, height, width = pixel_values.shape
        flat_pixels = pixel_values.reshape(batch_size * num_images, channels, height, width)
        with torch.no_grad() if not any(param.requires_grad for param in self.vision_encoder.parameters()) else contextlib.nullcontext():
            outputs = self.vision_encoder(pixel_values=flat_pixels, return_dict=True)
        if hasattr(outputs, "last_hidden_state"):
            image_tokens = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output"):
            image_tokens = outputs.pooler_output.unsqueeze(1)
        else:
            raise ValueError("Unsupported vision encoder output. Expected last_hidden_state or pooler_output.")
        return image_tokens.reshape(batch_size, -1, image_tokens.size(-1))

    def forward(self, bev_tensors, input_ids, attention_mask, labels=None, pixel_values=None):
        visual_tokens = self.encode_images(pixel_values)
        connector_tokens = self.connector(bev_tensors, visual_tokens=visual_tokens)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        prefix_mask = torch.ones(
            (input_ids.size(0), connector_tokens.size(1)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        inputs_embeds = torch.cat([connector_tokens, text_embeddings], dim=1)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        if labels is not None:
            label_prefix = torch.full(
                (labels.size(0), connector_tokens.size(1)),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([label_prefix, labels], dim=1)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
