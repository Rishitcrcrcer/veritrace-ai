"""Qwen2-VL-2B wrapper: 4-bit load, answer a question, capture attention in one pass.

This is the ONLY file that needs torch / transformers. Everything downstream works
on the numpy ModelRun it returns.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from ..core import ModelRun

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"


class Qwen2VLWrapper:
    def __init__(self, model_name: str = MODEL_NAME, min_pixels: int = 448 * 448,
                 max_pixels: int = 448 * 448, load_in_4bit: bool = True):
        kwargs = dict(
            torch_dtype=torch.bfloat16,        # fp16 gives NaN ("!!!!!") with this model
            device_map="auto",
            attn_implementation="eager",       # required, otherwise no attention weights
        )
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.tokenizer = self.processor.tokenizer

    # ------------------------------------------------------------------ inputs
    def _build_inputs(self, image: Image.Image, question: str):
        messages = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return inputs.to(self.model.device)

    def _strip_eos(self, gen_ids: torch.Tensor) -> torch.Tensor:
        eos = self.model.generation_config.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        ids = gen_ids[0].tolist()
        while ids and ids[-1] in eos:
            ids.pop()
        return torch.tensor([ids], device=gen_ids.device, dtype=gen_ids.dtype)

    # --------------------------------------------------------------- attention
    def _capture_attention(self, inputs, gen_ids) -> list[np.ndarray]:
        """One forward pass over prompt + answer (teacher forcing). Returns one
        head-averaged (L, L) float32 matrix per layer."""
        full_ids = torch.cat([inputs["input_ids"], gen_ids], dim=1)
        fwd = dict(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            output_attentions=True,
            use_cache=False,
        )
        if "mm_token_type_ids" in inputs:      # needed by newer transformers versions
            mm = inputs["mm_token_type_ids"]
            pad = torch.zeros(gen_ids.shape, dtype=mm.dtype, device=mm.device)
            fwd["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
        outputs = self.model(**fwd)
        return [a[0].float().mean(dim=0).cpu().numpy() for a in outputs.attentions]

    # --------------------------------------------------------------------- run
    @torch.no_grad()
    def run(self, image: Image.Image, question: str, max_new_tokens: int = 48) -> ModelRun:
        image = image.convert("RGB")
        inputs = self._build_inputs(image, question)
        P = inputs["input_ids"].shape[1]

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = self._strip_eos(out[:, P:])

        layer_attn = self._capture_attention(inputs, gen_ids)

        img_pos = torch.nonzero(inputs["input_ids"][0] == self.model.config.image_token_id)
        img_pos = img_pos.squeeze(1).cpu().numpy()
        t, h, w = inputs["image_grid_thw"][0].tolist()
        merge = self.model.config.vision_config.spatial_merge_size
        grid = (h // merge, w // merge)

        return ModelRun(
            answer_text=self.tokenizer.decode(gen_ids[0], skip_special_tokens=True),
            tokens=[self.tokenizer.decode([i]) for i in gen_ids[0].tolist()],
            layer_attn=layer_attn,
            prompt_len=P,
            img_pos=img_pos,
            grid=grid,
            image=image,
            question=question,
            model=self,
        )
