"""
QWENWrapper Exploration Script
================================

Compact exploration of the QWENWrapper class with helper utilities.
"""
import os
import torch
from torch import nn
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time


DTYPE = torch.bfloat16
MAX_SEQ_LEN = 4096
MAX_NEW_TOKS = 128
EOS_TOKS = (151643, 151645)


# ============================================================================
# Helper Functions
# ============================================================================

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


# ============================================================================
# QWENWrapper Class
# ============================================================================

class QWENWrapper(nn.Module):
    """Lightweight wrapper around the original HF model to prepare it for ONNX export.

    The wrapper preserves the original inference contract used by the exporter
    (inputs: past keys/values + input ids + lengths + attention mask) and
    produces the past keys/values updates plus the next-token id and kv_seq_len.
    """

    def __init__(self, qwen: nn.Module, max_seq_len: int, num_heads: int, num_key_value_heads: int, head_dim: int, num_layers: int):
        super().__init__()

        dtype = qwen.dtype
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
        
        try:
            device = next(self.qwen.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        data = self.qwen.model.embed_tokens.weight.data.to(device)
        zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
        scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
        embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
        
        self.register_buffer('zero_point', zero_point)
        self.register_buffer('scale', scale)
        self.register_buffer('embed_data', embed_data)
        
        position_ids = torch.arange(max_seq_len, device=device).unsqueeze(-1)
        idx_theta = position_ids * self.qwen.model.rotary_emb.inv_freq.to(device)
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.register_buffer('cos_rotary_pos_emb', torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).to(dtype=dtype))
        self.register_buffer('sin_rotary_pos_emb', torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).to(dtype=dtype))

        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8, device=device))) * -128)
        
        # # Pre-allocate large buffers for KV cache (avoid repeated allocations during generation)
        # # These will be reused across all forward passes
        # self.kv_buffer_keys = torch.zeros(num_layers, num_key_value_heads, 1, head_dim, max_seq_len, 
        #                                   dtype=torch.float32, device=device)
        # self.kv_buffer_values = torch.zeros(num_layers, num_key_value_heads, 1, max_seq_len, head_dim,
        #                                     dtype=torch.float32, device=device)

    def forward(self, key_buffer:torch.Tensor, value_buffer:torch.Tensor, 
                input_ids:torch.Tensor, attention_mask:torch.Tensor, past_len:int) -> torch.Tensor:
        """
        Args:
            key_buffer: pre-allocated tensor for key states [num_layers, num_kv_heads, 1, head_dim, max_seq_len]
            value_buffer: pre-allocated tensor for value states [num_layers, num_kv_heads, 1, max_seq_len, head_dim]
            input_ids: [batch=1, seq_len]
            past_len: scalar tensor
            attention_mask: tensor for attention mask [batch=1, seq_len]
        """
        
        # Use pre-allocated buffer for efficient in-place updates
        new_len = input_ids.size(1)
        end_idx = past_len + new_len
        assert end_idx <= key_buffer.size(-1), "Exceeded maximum sequence length in KV buffer"
        assert key_buffer.size(0) == self.num_layers, "Key buffer layer size mismatch"
        assert value_buffer.size(0) == self.num_layers, "Value buffer layer size mismatch"
        assert key_buffer.size(0) == value_buffer.size(0), "Key/Value buffer layer size mismatch"

        posemb_cos_q = self.cos_rotary_pos_emb[:, past_len:end_idx]
        posemb_sin_q = self.sin_rotary_pos_emb[:, past_len:end_idx]
        posemb_cos_k = posemb_cos_q.transpose(-1, -2)
        posemb_sin_k = posemb_sin_q.transpose(-1, -2)
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]

        attention_mask = self.attention_mask[:, :input_ids.size(1), :end_idx] * attention_mask
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        
        for i, layer in enumerate(self.qwen.model.layers): 
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim)
            new_k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim)
            new_v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = (layer.self_attn.q_norm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(0, 1)
            new_k = (layer.self_attn.k_norm.weight * (new_k / torch.sqrt(new_k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(2, 1, 3, 0)
            
            # Apply RoPE to k
            new_k = new_k * posemb_cos_k + rotate_half(new_k, self.head_dim_half, 2) * posemb_sin_k
            
            # Update KV cache using pre-allocated buffer (no concatenation needed)
            # Copy new data to buffer
            # key_buffer[i, :, :, :, past_len:end_idx] = new_k
            # value_buffer[i, :, :, past_len:end_idx, :] = new_v
            # Alternative using torch.narrow() and copy_()
            key_slice = key_buffer[i].narrow(-1, past_len, new_len)
            key_slice.copy_(new_k)

            value_slice = value_buffer[i].narrow(-2, past_len, new_len)
            value_slice.copy_(new_v)
            
            # Expand (zero-copy) instead of repeat
            # k_unique = key_buffer[i, :, :, :, :end_idx]
            # v_unique = value_buffer[i, :, :, :end_idx, :]
            k_unique = key_buffer[i].narrow(-1, 0, end_idx)  # [num_kv_heads, 1, head_dim, end_idx]
            v_unique = value_buffer[i].narrow(-2, 0, end_idx) # [num_kv_heads, 1, end_idx, head_dim]
            k_expanded = repeat_k(k_unique, self.num_key_value_groups, self.head_dim, self.num_heads)
            v_expanded = repeat_v(v_unique, self.num_key_value_groups, self.head_dim, self.num_heads)
            
            #TODO: use F.scaled_dot_product_attention() for better performance in torch
            attn = F.softmax(
                torch.matmul(q * posemb_cos_q + rotate_half(q, self.head_dim_half, -1) * posemb_sin_q, k_expanded) + attention_mask, dim=-1, dtype=q.dtype)
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
        tok_id = torch.argmax(logits, dim=-1, keepdim=True)
        return tok_id


def onnx_export(wrapper: QWENWrapper, output_path:str):
   
    device = wrapper.qwen.device
    # Export to ONNX
    print("Exporting to ONNX...")
    
    dummy_key_buffer = torch.zeros(wrapper.num_layers, wrapper.num_key_value_heads, 1, wrapper.head_dim, MAX_SEQ_LEN, 
                                   dtype=DTYPE, device=device)
    dummy_value_buffer = torch.zeros(wrapper.num_layers, wrapper.num_key_value_heads, 1, MAX_SEQ_LEN, wrapper.head_dim,
                                     dtype=DTYPE, device=device)
    dummy_input_ids = torch.zeros(1, 1, dtype=torch.int32, device=device)
    dummy_attention_mask = torch.zeros(1, 1, dtype=torch.int8, device=device)
    dummy_past_len = 0
    inputs = (dummy_key_buffer, dummy_value_buffer, dummy_input_ids, dummy_attention_mask, dummy_past_len)
    input_names = ['key_buffer', 'value_buffer', 'input_ids', 'attention_mask', 'past_len']
    output_names = ['next_token_id'] #TODO: full logits for sampling
    dynamic_axes = {
        'key_buffer': {4: 'max_seq_len'}, #[num_layers, num_kv_heads, 1, head_dim, *max_seq_len*]
        'value_buffer': {3: 'max_seq_len'}, #[num_layers, num_kv_heads, 1, *max_seq_len*, head_dim]
        'input_ids': {1: 'seq_len'}, #[batch=1, *seq_len*]
        'attention_mask': {1: 'seq_len'}, #[batch=1, *seq_len*]
        'past_len': {},
    }

    torch.onnx.export(
        wrapper,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        #dynamo=False,, dynamic_axes=dynamic_axes,
        dynamo=True, dynamic_shapes=dynamic_axes,
        #opset_version=18,
        #do_constant_folding=True,
    )

# ============================================================================
# Example Usage
# ============================================================================
def torch_inference(wrapper: QWENWrapper, tokenizer: AutoTokenizer):
    # Load config and model
   
    model = wrapper.qwen
    device = model.device
    
    # Your prompt with proper chat format
    prompt = "sing a song about the wonders of AI technology."
    
    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    print(f"\nPrompt: {prompt}")
    print(f"Formatted: {formatted_prompt}")
    print(f"Token IDs: {prompt_ids}\n")
    
    # Initialize cache (list of tensors per layer, like old version)
    num_layers = model.config.num_hidden_layers
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim

    key_buffer = torch.zeros(num_layers, num_key_value_heads, 1, head_dim, MAX_SEQ_LEN, 
                                          dtype=DTYPE, device=device)
    value_buffer = torch.zeros(num_layers, num_key_value_heads, 1, MAX_SEQ_LEN, head_dim,
                                            dtype=DTYPE, device=device)
        
    # Process prompt
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.int32, device=device)
    past_len = 0  # No history for first call
    attention_mask = torch.tensor([[0]], dtype=torch.int8, device=device)
    
    print("Processing prompt...")
    with torch.no_grad():
        next_token_tensor = wrapper(
            key_buffer, value_buffer, prompt_tensor, attention_mask, past_len
        )
 
    # Extract results
    next_token_id = next_token_tensor.item()
    
    print(f"Next predicted token ID: {next_token_id}")
    print(f"Token: '{tokenizer.decode([next_token_id])}'")
    
    # Generate more tokens with timing
    print(f"\nGenerating {MAX_NEW_TOKS} tokens with timing:\n")
    generated = prompt_ids + [next_token_id]

    start_time = time.time()
    for i in range(MAX_NEW_TOKS):
        next_id_tensor = torch.tensor([[next_token_id]], dtype=torch.int32, device=device)
        past_len = len(prompt_ids) + i  # Scalar integer for past length
        attention_mask = torch.tensor([[0]], dtype=torch.int8, device=device)
        
        with torch.no_grad():
            next_token_tensor = wrapper(
                key_buffer, value_buffer, next_id_tensor, attention_mask, past_len)

        
        next_token_id = next_token_tensor.item()
        generated.append(next_token_id)
        
        elapsed = time.time() - start_time
        tokens_generated = i + 1
        if i % 10 == 0:
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
            print(f"Token {i+1:2d}: '{tokenizer.decode([next_token_id]):20s}' | Time: {elapsed:6.2f}s | Speed: {tokens_per_sec:6.2f} tokens/sec")

        # Check if we should stop
        if next_token_id in EOS_TOKS:  # Stop tokens
            print("Stop token generated, ending.")
            break

    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = len(generated) - len(prompt_ids)
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\n{'='*80}")
    print(f"Generation Summary (NEW VERSION):")
    print(f"{'='*80}")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Speed: {avg_tokens_per_sec:.2f} tokens/sec")
    print(f"{'='*80}")
    
    print(f"\nFull generated text:")
    print(tokenizer.decode(generated))


def main():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open('config.json', 'r') as f:
        config = json.load(f)
    
    model_path = config['paths']['model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create wrapper
    print("Creating wrapper...")
    wrapper = QWENWrapper(
        qwen=model,
        max_seq_len=MAX_SEQ_LEN,
        num_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        num_layers=model.config.num_hidden_layers
    )
    wrapper.to(device)
    wrapper.eval()
    
    
    #torch_inference(wrapper, tokenizer)

    out_path = "qwen_wrapper_v2.onnx"
    onnx_export(wrapper, out_path)


if __name__ == "__main__":
    main()