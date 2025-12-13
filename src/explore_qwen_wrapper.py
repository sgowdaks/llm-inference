"""
QWENWrapper Exploration Script
================================

Compact exploration of the QWENWrapper class with helper utilities.
"""

import torch
from pathlib import Path


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

        position_ids = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(-1)
        idx_theta = position_ids * self.qwen.model.rotary_emb.inv_freq.to(device)
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.register_buffer('cos_rotary_pos_emb', torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half())
        self.register_buffer('sin_rotary_pos_emb', torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half())

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
            input_ids: [batch, seq_len]
            history_len: scalar tensor
            attention_mask: tensor for attention mask
        """
        
        # Use pre-allocated buffer for efficient in-place updates
        new_len = input_ids.size(1)
        end_idx = past_len + new_len
        assert end_idx <= key_buffer.size(-1), "Exceeded maximum sequence length in KV buffer"
        assert key_buffer.size(0) == self.num_layers, "Key buffer layer size mismatch"
        assert value_buffer.size(0) == self.num_layers, "Value buffer layer size mismatch"
        assert key_buffer.size(0) == value_buffer.size(0), "Key/Value buffer layer size mismatch"

        posemb_cos_q = self.cos_rotary_pos_emb[:, past_len:end_idx].float()
        posemb_sin_q = self.sin_rotary_pos_emb[:, past_len:end_idx].float()
        posemb_cos_k = posemb_cos_q.transpose(-1, -2)
        posemb_sin_k = posemb_sin_q.transpose(-1, -2)
        hidden_states = self.embed_data[input_ids].float() * self.scale[input_ids] + self.zero_point[input_ids]
        attention_mask = self.attention_mask[:, :input_ids.size(1), :end_idx] * attention_mask
        attention_mask = attention_mask.float()
        
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
            key_buffer[i, :, :, :, past_len:end_idx] = new_k
            value_buffer[i, :, :, past_len:end_idx, :] = new_v
            
            # Expand (zero-copy) instead of repeat
            k_expanded = repeat_k(key_buffer[i, :, :, :, :end_idx], self.num_key_value_groups, self.head_dim, self.num_heads)
            v_expanded = repeat_v(value_buffer[i, :, :, :end_idx, :], self.num_key_value_groups, self.head_dim, self.num_heads)
            
            import torch.nn.functional as F
            attn = F.softmax(torch.matmul(q * posemb_cos_q + rotate_half(q, self.head_dim_half, -1) * posemb_sin_q, k_expanded) + attention_mask, dim=-1, dtype=torch.float32)
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
        return tok_id


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import json
    
    # Load config and model
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    model_path = config['paths']['model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create wrapper
    print("Creating wrapper...")
    wrapper = QWENWrapper(
        qwen=model,
        max_seq_len=4096,
        num_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        num_layers=model.config.num_hidden_layers
    )
    wrapper.to(device)
    wrapper.eval()
    
    # Your prompt
    prompt = "sing a song about the wonders of AI technology."
    
    # Tokenize
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Token IDs: {prompt_ids}\n")
    
    # Initialize cache (list of tensors per layer, like old version)
    num_layers = model.config.num_hidden_layers
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    
    cache_keys = [torch.zeros(num_key_value_heads, 1, head_dim, 0, dtype=torch.float32, device=device) 
                  for _ in range(num_layers)]
    cache_values = [torch.zeros(num_key_value_heads, 1, 0, head_dim, dtype=torch.float32, device=device) 
                    for _ in range(num_layers)]
    
    # Process prompt
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.int32, device=device)
    history_len = torch.tensor([0], dtype=torch.int64, device=device)
    ids_len = torch.tensor([len(prompt_ids)], dtype=torch.int64, device=device)
    attention_mask = torch.tensor([0], dtype=torch.int8, device=device)
    
    print("Processing prompt...")
    with torch.no_grad():
        cache_keys, cache_values, next_token_tensor, kv_seq_len = wrapper(
            cache_keys, cache_values, prompt_tensor, history_len, ids_len, attention_mask
        )
    
    # Extract results
    next_token_id = next_token_tensor.item()
    
    print(f"Next predicted token ID: {next_token_id}")
    print(f"Token: '{tokenizer.decode([next_token_id])}'")
    
    # Generate more tokens with timing
    print("\nGenerating 20 tokens with timing:\n")
    generated = prompt_ids + [next_token_id]
    
    import time
    start_time = time.time()
    
    for i in range(20):
        next_id_tensor = torch.tensor([[next_token_id]], dtype=torch.int32, device=device)
        history_len = torch.tensor([len(prompt_ids) + i], dtype=torch.int64, device=device)
        ids_len = torch.tensor([1], dtype=torch.int64, device=device)
        attention_mask = torch.tensor([0], dtype=torch.int8, device=device)
        
        with torch.no_grad():
            cache_keys, cache_values, next_token_tensor, kv_seq_len = wrapper(
                cache_keys, cache_values, next_id_tensor, history_len, ids_len, attention_mask
            )
        
        next_token_id = next_token_tensor.item()
        generated.append(next_token_id)
        
        elapsed = time.time() - start_time
        tokens_generated = i + 1
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
        
        print(f"Token {i+1:2d}: '{tokenizer.decode([next_token_id]):20s}' | Time: {elapsed:6.2f}s | Speed: {tokens_per_sec:6.2f} tokens/sec")
    
    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = 20
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

