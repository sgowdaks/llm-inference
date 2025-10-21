"""ONNX inference helper for exported Qwen models.

This script is a cleaned-up, more maintainable rewrite of the original
example (from DakeQQ/Native-LLM-for-Android). It keeps the same
behaviour but improves readability, adds type hints, structured logging,
argument parsing, safer file handling and small sanity checks.

Notes:
- The script intentionally keeps external behaviour of the original:
  loading tokenizer, reading config.json, using onnxruntime OrtValues,
  and decoding tokens until a stop token.
- It does not add external dependencies beyond `transformers`,
  `onnxruntime` and `numpy` which the original already required.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    model_dir: Path
    onnx_file: Path
    model_config: Path


DEFAULT_STOP_TOKENS = [151643, 151645]
DEFAULT_MAX_SEQ_LEN = 4096


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def ort_device_name(use_cuda: bool) -> str:
    return "cuda" if use_cuda else "cpu"


def build_session(onnx_path: Path, prefer_cuda: bool = True) -> Tuple[onnxruntime.InferenceSession, bool]:
    """Create and configure an ONNX Runtime session for CPU inference.

    The session options mirror the original example but are expressed in
    a clearer and commented way.
    """
    session_opts = onnxruntime.SessionOptions()
    # Logging
    session_opts.log_severity_level = 4
    session_opts.log_verbosity_level = 4

    # Threading: 0 means let ORT choose reasonable defaults
    session_opts.inter_op_num_threads = 0
    session_opts.intra_op_num_threads = 0

    # Memory / optimization options
    session_opts.enable_cpu_mem_arena = True
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Session config entries copied from the original example.
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
    session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
    session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    # Keep disabling sync of execution providers if available in this ORT build
    session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
    session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
    session_opts.add_session_config_entry(
        "session.use_device_allocator_for_initializers",
        "1",
    )

    # Decide which providers to use. Prefer CUDA if available and requested.
    available = onnxruntime.get_available_providers()
    use_cuda = False
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        use_cuda = True
    else:
        providers = ["CPUExecutionProvider"]

    logger.debug("Creating ONNX Runtime session for %s with providers=%s", onnx_path, providers)
    sess = onnxruntime.InferenceSession(str(onnx_path), sess_options=session_opts, providers=providers)
    return sess, use_cuda


def prepare_inputs(
    tokenizer,
    prompt: str,
    num_key_value_heads: int,
    head_dim: int,
    num_layers: int,
    use_cuda: bool = False,
) -> Dict[str, onnxruntime.OrtValue]:
    """Tokenize prompt and build initial OrtValue inputs used by the model.

    Returns a mapping from input name index positions to OrtValue objects.
    """
    tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)

    device_name = ort_device_name(use_cuda)
    device_id = 0  # change if you want a different GPU

    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_name, device_id)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(
        np.array([tokens.shape[-1]], dtype=np.int64), device_name, device_id
    )
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_name, device_id)
    attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_name, device_id)

    past_keys = onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros((num_key_value_heads, 1, head_dim, 0), dtype=np.float32), device_name, device_id
    )
    past_values = onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros((num_key_value_heads, 1, 0, head_dim), dtype=np.float32), device_name, device_id
    )


    inputs = {
        "input_ids": input_ids,
        "ids_len": ids_len,
        "history_len": history_len,
        "attention_mask": attention_mask,
        "past_keys": past_keys,
        "past_values": past_values,
    }
    return inputs


def run_decode_loop(
    sess: onnxruntime.InferenceSession,
    tokenizer,
    inputs: Dict[str, onnxruntime.OrtValue],
    input_names: List[str],
    output_names: List[str],
    stop_tokens: List[int],
    max_decode: int = 512,
    short_answer: bool = False,
) -> None:
    """Run the autoregressive decode loop and print tokens as they are generated.

    This keeps the original behaviour: it updates inputs with outputs each step
    and decodes the next token until a stop token is produced or max_decode
    is reached.
    """
    # Map the provided high-level inputs into the positional input names the session expects.
    # The original script placed the last 4 inputs as input ids, history_len, ids_len, attention_mask.
    # We follow the same mapping here: use the tail names for these values, and the head names
    # for the past key/values.

    amount_of_outputs = len(output_names)

    # Build initial input_feed mapping from input names expected by the session
    input_feed: Dict[str, onnxruntime.OrtValue] = {}

    # We expect the session to expose some inputs where the last 4 correspond to ids/lengths/attention.
    # If that assumption is violated, we fall back to a best-effort mapping by name.
    try:
        # map last 4 inputs by position
        input_feed[input_names[-4]] = inputs["input_ids"]
        input_feed[input_names[-3]] = inputs["history_len"]
        input_feed[input_names[-2]] = inputs["ids_len"]
        input_feed[input_names[-1]] = inputs["attention_mask"]

        # map past keys and values to the front inputs
        num_inputs = len(input_names)
        num_layers = (num_inputs - 4) // 2
        for i in range(num_layers):
            input_feed[input_names[i]] = inputs["past_keys"]
        for i in range(num_layers, num_layers * 2):
            input_feed[input_names[i]] = inputs["past_values"]

    except Exception:  # pragma: no cover - defensive fallback
        logger.warning("Unexpected input layout; falling back to name-based mapping")
        # name-based mapping
        for name in input_names:
            lname = name.lower()
            if "input_ids" in lname:
                input_feed[name] = inputs["input_ids"]
            elif "history_len" in lname or "history" in lname:
                input_feed[name] = inputs["history_len"]
            elif "ids_len" in lname or "ids" in lname:
                input_feed[name] = inputs["ids_len"]
            elif "attention" in lname:
                input_feed[name] = inputs["attention_mask"]
            elif "past_key" in lname:
                input_feed[name] = inputs["past_keys"]
            elif "past_value" in lname:
                input_feed[name] = inputs["past_values"]

    logger.info("Starting decode loop (max tokens=%d)", max_decode)

    # Determine device for small OrtValue creations (fallback to cpu if session lacks CUDA)
    providers = sess.get_providers()
    device_name_for_ort = "cuda" if "CUDAExecutionProvider" in providers else "cpu"

    decoded_tokens: List[str] = []
    num_decoded = 0
    start_time = time.time()
    decoded_text = ""

    while num_decoded < max_decode:
        all_outputs = sess.run_with_ort_values(output_names, input_feed)

        # The model's penultimate output is the token logits/ids in the original script.
        # We defensively search for a suitable output that looks like token ids.
        potential_token = None
        try:
            potential_token = onnxruntime.OrtValue.numpy(all_outputs[-2])
        except Exception:
            # try to find a 2D int array among outputs
            for out in all_outputs:
                try:
                    arr = onnxruntime.OrtValue.numpy(out)
                    if arr.ndim == 2 and arr.dtype == np.int64 or arr.dtype == np.int32:
                        potential_token = arr
                        break
                except Exception:
                    continue

        if potential_token is None:
            logger.error("Could not determine token output from model outputs")
            break

        max_logit_ids = potential_token
        num_decoded += 1

        # check stop token
        if int(max_logit_ids.flatten()[0]) in stop_tokens:
            logger.info("Encountered stop token after %d tokens", num_decoded)
            break

        # update inputs with all outputs so next step uses updated past key/values
        for i, name in enumerate(input_names):
            try:
                input_feed[name] = all_outputs[i]
            except Exception:
                # ignore missing outputs
                pass

        # for the first token, certain flags are reset as in the original script
        if num_decoded < 2:
            input_feed[input_names[-1]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                np.array([0], dtype=np.int8), device_name_for_ort, 0
            )
            input_feed[input_names[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                np.array([1], dtype=np.int64), device_name_for_ort, 0
            )

        # decode and print the token
        try:
            token_str = tokenizer.decode(max_logit_ids[0])
        except Exception:
            token_str = str(max_logit_ids.flatten()[0])

        print(token_str, end="", flush=True)
        decoded_tokens.append(token_str)
        decoded_text += token_str

        # Short-answer heuristics: stop early when requested
        if short_answer:
            # 1) numeric detection (simple integer match)
            import re

            m = re.search(r"\b-?\d+\b", decoded_text)
            if m and len(m.group(0)) <= 4:
                logger.debug("Short-answer numeric detected: %s", m.group(0))
                break

            # 2) stop on first sentence terminator/newline if text is short
            if any(sep in decoded_text for sep in ("\n", ".", "?", "!")) and len(decoded_text) < 128:
                logger.debug("Short-answer sentence terminator detected; stopping early")
                break

    elapsed = time.time() - start_time
    rate = num_decoded / elapsed if elapsed > 0 else 0.0
    print(f"\n\nDecode: {rate:.3f} token/s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run ONNX exported Qwen model for inference")
    parser.add_argument("--config", type=Path, default=Path("./config.json"), help="Path to config.json")
    parser.add_argument("--test-mode", default=True, help="Run example tests from test_file in config")
    parser.add_argument("--prompt", type=str, default="Hello there, how are u?", help="Prompt to run when not in test mode")
    parser.add_argument("--max-decode", type=int, default=512, help="Maximum tokens to decode")
    parser.add_argument("--short-answer", default=False, help="Stop early and return a short answer (numeric or first sentence)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = args.config
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        return 2

    data = load_json(cfg_path)

    paths = data.get("paths", {})
    model_dir = Path(paths.get("model_path", ""))
    onnx_file = Path(paths.get("onnx_file", ""))
    model_config = Path(paths.get("model_config", ""))

    if not model_dir.exists():
        logger.warning("Model directory does not exist: %s", model_dir)

    if not onnx_file.exists():
        logger.error("ONNX model file does not exist: %s", onnx_file)
        return 3

    if not model_config.exists():
        logger.error("Model config file does not exist: %s", model_config)
        return 4

    # load tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    cfg = load_json(model_config)

    num_key_value_heads = int(cfg.get("num_key_value_heads", 0))
    head_dim = int(cfg.get("head_dim", 0))
    num_layers = int(cfg.get("num_hidden_layers", 0))

    sess, use_cuda = build_session(onnx_file)

    # extract input and output names once
    input_meta = sess.get_inputs()
    output_meta = sess.get_outputs()
    input_names = [m.name for m in input_meta]
    output_names = [m.name for m in output_meta]

    if args.test_mode:
        test_file = paths.get("test_file")
        if not test_file:
            logger.error("test_file not specified in config but --test-mode was set")
            return 5
        test_path = Path(test_file)
        if not test_path.exists():
            logger.error("test file not found: %s", test_path)
            return 6
        tests = load_json(test_path)
        for t in tests:
            prompt = t.get("prompt", "")
            print(f"\n\nTest Question: {prompt}\nQwen Answering:\n")
            inputs = prepare_inputs(tokenizer, prompt, num_key_value_heads, head_dim, num_layers, use_cuda=use_cuda)
            run_decode_loop(sess, tokenizer, inputs, input_names, output_names, DEFAULT_STOP_TOKENS, max_decode=args.max_decode, short_answer=args.short_answer)
    else:
        prompt = args.prompt
        print(f"\n\nPrompt: {prompt}\nQwen Answering:\n")
        inputs = prepare_inputs(tokenizer, prompt, num_key_value_heads, head_dim, num_layers, use_cuda=use_cuda)
    run_decode_loop(sess, tokenizer, inputs, input_names, output_names, DEFAULT_STOP_TOKENS, max_decode=args.max_decode, short_answer=args.short_answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())