"""Exporter and test-runner for Qwen models.

This script is a cleaned-up, more maintainable rewrite of the original
example (from DakeQQ/Native-LLM-for-Android). It keeps the same
behaviour but improves readability, adds type hints, structured logging,
argument parsing, safer file handling and small sanity checks.


This module contains two main flows:
- export: build an optimized torch.nn.Module wrapper and export to ONNX
- test: run prompts from a test JSON file directly with Hugging Face model

The new interface is a small CLI that lets you choose the backend and
keeps the original export behaviour intact.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    model_path: Path
    onnx_path: Path
    model_config: Path
    max_seq_len: int = 4096


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def quantize_to_uint8(tensor: torch.Tensor, scale: float, zero_point: torch.Tensor) -> torch.Tensor:
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def rotate_half(x: torch.Tensor, head_dim_half: int, dim: int) -> torch.Tensor:
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states: torch.Tensor, num_key_value_groups: int, head_dim: int, num_heads: int) -> torch.Tensor:
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states: torch.Tensor, num_key_value_groups: int, head_dim: int, num_heads: int) -> torch.Tensor:
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


class QWENWrapper(torch.nn.Module):
    """Lightweight wrapper around the original HF model to prepare it for ONNX export.

    The wrapper preserves the original inference contract used by the exporter
    (inputs: past keys/values + input ids + lengths + attention mask) and
    produces the past keys/values updates plus the next-token id and kv_seq_len.
    """

    def __init__(self, qwen, max_seq_len: int, num_heads: int, num_key_value_heads: int, head_dim: int, num_layers: int):
        super().__init__()
        self.qwen = qwen
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.qwen.model.layers._modules[f"{i}"].self_attn.q_norm.weight.data *= scale_factor
            self.qwen.model.layers._modules[f"{i}"].self_attn.k_norm.weight.data *= scale_factor
        # Register constant tensors as buffers so they move with .to(device)
        # determine the device used by the HF model and create buffers on that device
        try:
            device = next(self.qwen.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        data = self.qwen.model.embed_tokens.weight.data.to(device)
        zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
        embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
        # buffers won't be considered parameters and will follow module.to(device)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('scale', scale)
        self.register_buffer('embed_data', embed_data)

        # create position/rotary buffers on the same device as the model
        position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(-1)
        idx_theta = position_ids * self.qwen.model.rotary_emb.inv_freq.to(device)
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.register_buffer('cos_rotary_pos_emb', torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half())
        self.register_buffer('sin_rotary_pos_emb', torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half())

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        # attention mask as buffer so it moves to the device with the module
        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8, device=device))) * -128)

    def forward(self, *all_inputs):
        input_ids = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2)
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.qwen.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = (layer.self_attn.q_norm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(0, 1)
            k = (layer.self_attn.k_norm.weight * (k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(2, 1, 3, 0)
            k = torch.cat((all_inputs[i], k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, 2) * rotary_pos_emb_sin_k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads)
            attn = torch.nn.functional.softmax(torch.matmul(q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.qwen.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        return *self.save_key, *self.save_value, torch.argmax(self.qwen.lm_head(hidden_states), dim=-1, keepdim=True).int(), kv_seq_len


def export_to_onnx(config: ExportConfig) -> None:
    logger.info("Starting export to ONNX: %s -> %s", config.model_path, config.onnx_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model on CPU first and move to device to avoid accidental device_map issues
    model = AutoModelForCausalLM.from_pretrained(str(config.model_path), torch_dtype=torch.float32, trust_remote_code=True, low_cpu_mem_usage=True).eval()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)

    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads

    wrapper = QWENWrapper(model, config.max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers)
    # move wrapper buffers and module to target device so buffers (embed_data, rotary, attention_mask) follow
    wrapper.to(device)

    # generate dummy inputs
    # create dummy inputs on the correct device
    attention_mask = torch.tensor([0], dtype=torch.int8, device=device)
    ids_len = torch.tensor([10], dtype=torch.int64, device=device)   # dummy
    input_ids = torch.ones((1, int(ids_len.item())), dtype=torch.int32, device=device)
    history_len = torch.zeros(1, dtype=torch.int64, device=device)
    past_keys = torch.zeros((num_key_value_heads, 1, head_dim, 0), dtype=torch.float32, device=device)
    past_values = torch.zeros((num_key_value_heads, 1, 0, head_dim), dtype=torch.float32, device=device)

    all_inputs: List[torch.Tensor] = []
    input_names: List[str] = []
    output_names: List[str] = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}

    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {3: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'history_len_plus_ids_len'}

    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}

    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('max_logit_id')
    output_names.append('kv_seq_len')

    # Export: ensure wrapper is on device (buffers moved) and inputs are on same device
    torch.onnx.export(
        wrapper,
        tuple(all_inputs),
        str(config.onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=13,  # Changed from 17 to 13 for better compatibility
    )


def run_hf_tests(model_path: Path, test_file: Path, short_answer: bool = False) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True).eval()
    model.to(device)

    tests = load_json(test_file)
    for t in tests:
        prompt = t.get('prompt', '')
        print(f"\n\nTest Question: {prompt}\nHF Answering:\n")
        # simple greedy generation
        # Request an explicit attention mask (padding) to avoid ambiguous pad/eos behavior
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        # move runtime inputs to the same device as the model
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        input_len = input_ids.shape[1]

        # ensure we have sensible eos/pad ids so generation can stop early
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        # generate only a small number of tokens for short factual answers and use greedy decoding
        out = model.generate(
            input_ids.to(device),
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            num_return_sequences=1,
        )

        # decode only the newly generated tokens (avoid prompt overlap and repeated prompts)
        generated_ids = out[0][input_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        answer = decoded
        if short_answer:
            # take first sentence
            for sep in ('\n', '.', '?', '!'):
                if sep in answer:
                    answer = answer.split(sep)[0].strip()
                    break
        print(answer)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Export or test Qwen models')
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--mode', choices=('export', 'test'), default='export')
    parser.add_argument('--short-answer', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    cfg = load_json(args.config)
    paths = cfg.get('paths', {})
    model_path = Path(paths.get('model_path', ''))
    onnx_path = Path(paths.get('onnx_file', ''))
    model_config = Path(paths.get('model_config', ''))

    if args.mode == 'export':
        export_cfg = ExportConfig(model_path=model_path, onnx_path=onnx_path, model_config=model_config)
        export_to_onnx(export_cfg)
        print("Export completed successfully.")
    else:
        test_file = Path(paths.get('test_file'))
        if not test_file.exists():
            logger.error('test file not found: %s', test_file)
            return 2
        run_hf_tests(model_path, test_file, short_answer=args.short_answer)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())