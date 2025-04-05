# model_service.py (Final nested approach, no flattening)

import os
import sys
import time
import torch
import torch.nn as nn
import torch_neuronx
import grpc
from concurrent import futures
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

from protos import model_service_pb2, model_service_pb2_grpc

def create_empty_past(num_layers, num_kv, past_seq, head_dim):
    """
    Build a *nested* tuple of length num_layers, where each item is (k, v).
    Each (k, v) shaped [batch_size=1, num_kv, past_seq, head_dim].
    This must match how we traced the model.
    """
    dummy_past_list = []
    for _ in range(num_layers):
        k = torch.zeros((1, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
        v = torch.zeros((1, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
        dummy_past_list.append((k, v))
    # Return a nested tuple: ((k1,v1),(k2,v2), ...)
    return tuple(dummy_past_list)

class UserConfigManager:
    def __init__(self):
        self.configs = {}
    def set_user_config(self, user_uuid, max_length, gamma, exact_mode, debug_mode):
        self.configs[user_uuid] = (max_length, gamma, exact_mode, debug_mode)
    def get_user_config(self, user_uuid):
        return self.configs.get(user_uuid, (128, 4, True, False))

# This wrapper ensures the model returns (logits, pkv) as a 2-tuple, avoiding dict outputs.
class LlamaKVWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        # Return (logits, pkv) instead of a dict
        return (outputs.logits, outputs.past_key_values)

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self, model_path, compiled_model_path=None, compile_model=False, max_context=2048):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.config_manager = UserConfigManager()

        print(f"[Target] Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        base_model = None
        base_config = None

        if compiled_model_path and os.path.isfile(compiled_model_path):
            # Load a precompiled model
            print(f"[Target] Loading precompiled model from {compiled_model_path}")
            self.model = torch.jit.load(compiled_model_path)
        else:
            # Load HF model from disk
            print(f"[Target] Loading HF model {model_path} in BF16 for Trainium.")
            base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            base_model.eval()
            base_config = base_model.config

            if compile_model:
                print("[Target] Compiling model with torch_neuronx.trace...")
                wrapper = LlamaKVWrapper(base_model)

                num_layers = base_config.num_hidden_layers
                num_kv = getattr(base_config, "num_key_value_heads", base_config.num_attention_heads)
                head_dim = getattr(base_config, "head_dim", base_config.hidden_size // num_kv)
                batch_size = 1
                seq_len   = 1
                past_seq  = max_context - 1

                # Build a *nested* past: ((k,v),(k,v),...)
                dummy_past_list = []
                for _ in range(num_layers):
                    k = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    v = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    dummy_past_list.append((k, v))
                dummy_past = tuple(dummy_past_list)

                dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

                self.model = torch_neuronx.trace(wrapper, (dummy_input_ids, dummy_past))
                if compiled_model_path:
                    print(f"[Target] Saving compiled model to {compiled_model_path}")
                    self.model.save(compiled_model_path)
            else:
                # CPU fallback
                self.model = base_model

        # If we do have base_model, keep base_config. Otherwise fallback
        if base_model is not None:
            base_config = base_model.config

        if base_config is not None:
            num_layers = base_config.num_hidden_layers
            num_kv = getattr(base_config, "num_key_value_heads", base_config.num_attention_heads)
            head_dim = getattr(base_config, "head_dim", base_config.hidden_size // num_kv)
            past_seq = max_context - 1
        else:
            print("[Target] No base_config found; using fallback shapes (16-layer, 8-kv, head_dim=64).")
            num_layers = 16
            num_kv = 8
            head_dim = 64
            past_seq = max_context - 1

        # Build a nested "empty" past
        self.empty_past = create_empty_past(num_layers, num_kv, past_seq, head_dim)

        # Optionally move trace to device
        try:
            torch_neuronx.experimental.set_neuron_cores([0])
            torch_neuronx.move_trace_to_device(self.model, 0)
        except Exception as e:
            print(f"[Target] Could not move model to Neuron device: {e}")

        print("[Target] ModelServiceServicer init complete.")

    def PrepareSpeculative(self, request, context):
        """
        Multi-token decode using the traced forward pass (no .generate()).
        We'll generate up to 'request.generate_step' new tokens.
        """
        prompt_text = request.prompt
        max_new_tokens = request.generate_step

        # Convert prompt -> input IDs
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids

        # Start with a nested "empty" past
        past_key_values = self.empty_past

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Two positional args
                logits, past_key_values = self.model(input_ids, past_key_values)
                # pick top token
                next_token_id = torch.argmax(logits[..., -1, :], dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        first_tokens_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return model_service_pb2.PrepareSpeculativeResponse(first_tokens=first_tokens_text)

    def VerifyTokens(self, request, context):
        return model_service_pb2.VerifyTokensResponse(verified=True)

    def GenerateContent(self, request, context):
        return model_service_pb2.GenerateContentResponse(generated_text="Not implemented")

def serve(port, model_path, compiled_model_path, compile_model, max_context):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = ModelServiceServicer(model_path, compiled_model_path, compile_model, max_context)
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[Target] gRPC server listening on port {port}, model={model_path}")
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/models/llama-3.2-1b")
    parser.add_argument("--compiled-model-path", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-context", type=int, default=2048)
    args = parser.parse_args()

    serve(args.port, args.model_path, args.compiled_model_path, args.compile, args.max_context)