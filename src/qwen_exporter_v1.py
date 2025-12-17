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
    # Use expand instead of repeat for zero-copy operation
    return kv_states.unsqueeze(1).expand(-1, num_key_value_groups, -1, -1, -1).reshape(num_heads, head_dim, -1)


def repeat_v(kv_states: torch.Tensor, num_key_value_groups: int, head_dim: int, num_heads: int) -> torch.Tensor:
    # Use expand instead of repeat for zero-copy operation
    return kv_states.unsqueeze(1).expand(-1, num_key_value_groups, -1, -1, -1).reshape(num_heads, -1, head_dim)


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
            # Detach and clone to avoid gradient tracking
            self.qwen.model.layers._modules[f"{i}"].self_attn.q_norm.weight.data = (self.qwen.model.layers._modules[f"{i}"].self_attn.q_norm.weight.data * scale_factor).detach()
            self.qwen.model.layers._modules[f"{i}"].self_attn.k_norm.weight.data = (self.qwen.model.layers._modules[f"{i}"].self_attn.k_norm.weight.data * scale_factor).detach()
        # Register constant tensors as buffers so they move with .to(device)
        # determine the device used by the HF model and create buffers on that device
        try:
            device = next(self.qwen.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # Detach from computation graph to avoid requires_grad issues during ONNX export
        data = self.qwen.model.embed_tokens.weight.data.detach().to(device)
        zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
        embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
        # buffers won't be considered parameters and will follow module.to(device)
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('scale', scale)
        self.register_buffer('embed_data', embed_data)

        # create position/rotary buffers on the same device as the model
        # Detach inv_freq from computation graph
        position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(-1)
        idx_theta = position_ids * self.qwen.model.rotary_emb.inv_freq.detach().to(device)
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        # Detach rotary embeddings before registering as buffers
        self.register_buffer('cos_rotary_pos_emb', torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half().detach())
        self.register_buffer('sin_rotary_pos_emb', torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half().detach())

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        # attention mask as buffer so it moves to the device with the module
        # This is already a new tensor so shouldn't need detach, but add for safety
        self.register_buffer('attention_mask', ((1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8, device=device))) * -128).detach())
        
        # Pre-allocate large buffers for KV cache (avoid repeated allocations during generation)
        # These will be reused across all forward passes
        # self.kv_buffer_keys = torch.zeros(num_layers, num_key_value_heads, 1, head_dim, max_seq_len, 
        #                                   dtype=torch.float32, device=device)
        # self.kv_buffer_values = torch.zeros(num_layers, num_key_value_heads, 1, max_seq_len, head_dim,
        #                                     dtype=torch.float32, device=device)

    def forward(self, key_buffer:torch.Tensor, value_buffer:torch.Tensor, 
                input_ids:torch.Tensor, attention_mask:torch.Tensor, past_len:torch.Tensor) -> torch.Tensor:
        """
        Args:
            key_buffer: pre-allocated tensor for key states [num_layers, num_kv_heads, 1, head_dim, max_seq_len]
            value_buffer: pre-allocated tensor for value states [num_layers, num_kv_heads, 1, max_seq_len, head_dim]
            input_ids: [batch, seq_len]
            past_len: scalar tensor with past sequence length
            attention_mask: tensor for attention mask
        """
        
        # Use narrow operation instead of slicing to avoid converting past_len to Python scalar
        # This allows ONNX to properly handle dynamic past_len values
        new_len = input_ids.size(1)
        
        # For rotary embeddings, we need to select the appropriate slice
        # Use narrow which works with tensor indices in ONNX
        past_len_val = past_len.squeeze() if past_len.dim() > 0 else past_len
        
        # Get positional embeddings for current positions
        # We'll use index_select which is ONNX-compatible
        indices = torch.arange(new_len, device=input_ids.device, dtype=torch.long) + past_len_val
        posemb_cos_q = torch.index_select(self.cos_rotary_pos_emb, 1, indices).float()
        posemb_sin_q = torch.index_select(self.sin_rotary_pos_emb, 1, indices).float()
        posemb_cos_k = posemb_cos_q.transpose(-1, -2)
        posemb_sin_k = posemb_sin_q.transpose(-1, -2)
        
        hidden_states = self.embed_data[input_ids].float() * self.scale[input_ids] + self.zero_point[input_ids]
        
        # Build attention mask dynamically based on past_len
        end_idx_val = past_len_val + new_len
        mask_indices = torch.arange(end_idx_val, device=input_ids.device, dtype=torch.long)
        selected_mask = torch.index_select(self.attention_mask, 2, mask_indices)
        selected_mask = selected_mask[:, :new_len, :]
        attention_mask_expanded = selected_mask * attention_mask
        attention_mask_expanded = attention_mask_expanded.float()
        
        for i, layer in enumerate(self.qwen.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim)
            new_k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim)
            new_v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = (layer.self_attn.q_norm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(0, 1)
            new_k = (layer.self_attn.k_norm.weight * (new_k / torch.sqrt(new_k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(2, 1, 3, 0)
            
            # Apply RoPE to k
            new_k = new_k * posemb_cos_k + rotate_half(new_k, self.head_dim_half, 2) * posemb_sin_k
            
            # Update KV cache - scatter new keys/values into buffer
            # Create index tensor for scatter operation
            kv_start_idx = past_len_val.long()
            
            # For keys: need to update dimension 4 (seq_len dimension)
            # Reshape new_k to match: [num_kv_heads, 1, head_dim, new_len]
            # Scatter into key_buffer[i] at positions [kv_start_idx:kv_start_idx+new_len]
            for seq_idx in range(new_len):
                actual_idx = kv_start_idx + seq_idx
                key_buffer[i, :, :, :, actual_idx] = new_k[:, :, :, seq_idx]
                value_buffer[i, :, :, actual_idx, :] = new_v[:, :, seq_idx, :]
            
            # Expand (zero-copy) instead of repeat
            # Extract KV cache up to current position
            end_pos = (past_len_val + new_len).long()
            k_cache = key_buffer[i, :, :, :, :end_pos]
            v_cache = value_buffer[i, :, :, :end_pos, :]
            k_expanded = repeat_k(k_cache, self.num_key_value_groups, self.head_dim, self.num_heads)
            v_expanded = repeat_v(v_cache, self.num_key_value_groups, self.head_dim, self.num_heads)
            
            import torch.nn.functional as F
            attn = F.softmax(torch.matmul(q * posemb_cos_q + rotate_half(q, self.head_dim_half, -1) * posemb_sin_q, k_expanded) + attention_mask_expanded, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v_expanded).transpose(0, 1).contiguous().view(1, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        
        hidden_states = hidden_states[:, -1]
        hidden_states = self.qwen.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        
        logits = self.qwen.lm_head(hidden_states)
        # TODO: return full logits for sampling
        tok_id = torch.argmax(logits, dim=-1, keepdim=True).int()
        
        # Return token id, updated buffers, and new sequence length
        new_seq_len = (past_len_val + new_len).unsqueeze(0).long()
        return tok_id, key_buffer, value_buffer, new_seq_len

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
    wrapper.eval()
    
    # Disable gradient tracking for all parameters to prevent requires_grad issues during ONNX export
    for param in wrapper.parameters():
        param.requires_grad_(False)

    # generate dummy inputs for new signature
    # create dummy inputs on the correct device
    attention_mask = torch.tensor([0], dtype=torch.int8, device=device)
    ids_len = 10  # dummy sequence length
    input_ids = torch.ones((1, ids_len), dtype=torch.int32, device=device)
    past_len = 0  # scalar integer for past length
    
    # Pre-allocated KV buffers (full max_seq_len size)
    key_buffer = torch.zeros((num_layers, num_key_value_heads, 1, head_dim, config.max_seq_len), 
                             dtype=torch.float32, device=device)
    value_buffer = torch.zeros((num_layers, num_key_value_heads, 1, config.max_seq_len, head_dim),
                               dtype=torch.float32, device=device)

    all_inputs: List[torch.Tensor] = []
    input_names: List[str] = []
    output_names: List[str] = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}

    # Single key buffer input
    input_names.append('key_buffer')
    all_inputs.append(key_buffer)
    dynamic_axes['key_buffer'] = {4: 'max_seq_len'}  # max_seq_len dimension is fixed
    
    # Single value buffer input
    input_names.append('value_buffer')
    all_inputs.append(value_buffer)
    dynamic_axes['value_buffer'] = {3: 'max_seq_len'}  # max_seq_len dimension is fixed

    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    input_names.append('past_len')
    all_inputs.append(torch.tensor([past_len], dtype=torch.int64, device=device))
    
    # Output names matching the return signature
    output_names.append('next_token_id')
    output_names.append('updated_key_buffer')
    output_names.append('updated_value_buffer')
    output_names.append('new_seq_len')
    
    # Add dynamic axes for output buffers
    dynamic_axes['updated_key_buffer'] = {4: 'max_seq_len'}
    dynamic_axes['updated_value_buffer'] = {3: 'max_seq_len'}

    # Export: ensure wrapper is on device (buffers moved) and inputs are on same device
    # Use dynamo=False to force legacy TorchScript export (PyTorch 2.9+ defaults to new API)
    torch.onnx.export(
        wrapper,
        tuple(all_inputs),
        str(config.onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=13,  # Changed from 17 to 13 for better compatibility
        dynamo=False,  # Force legacy export mode
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