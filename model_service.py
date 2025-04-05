# model_service.py (Updated to avoid dictionary in tracer output)

import os
import sys
import time
import torch
import torch.nn as nn
import torch_neuronx
import grpc
from concurrent import futures

from transformers import AutoModelForCausalLM, AutoTokenizer
from protos import model_service_pb2, model_service_pb2_grpc
from collections import defaultdict

class UserConfigManager:
    def __init__(self):
        self.configs = {}
    def set_user_config(self, user_uuid, max_length, gamma, exact_mode, debug_mode):
        self.configs[user_uuid] = (max_length, gamma, exact_mode, debug_mode)
    def get_user_config(self, user_uuid):
        return self.configs.get(user_uuid, (128, 4, True, False))

# Instead of returning a dict with "logits" and "past_key_values", we'll
# manually produce a 2-tuple, so JIT sees a single typed output (Tuple[Tensor, Tuple]).
class LlamaKVWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, past_key_values=None):
        # HF model returns a 'CausalLMOutputWithPast', which is effectively a dict
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        # Instead of returning outputs (which is a dict with "logits" and "past_key_values"),
        # we return a 2-tuple. The 1st element is the logits tensor, 2nd is the tuple of (k, v).
        # This avoids "Tracer cannot infer type of {dict with mixed types}".
        return (outputs.logits, outputs.past_key_values)

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self, model_path, compiled_model_path=None, compile_model=False, max_context=2048):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.user_data = defaultdict(lambda: {
            "tokens": "", "drafts": []
        })
        self.config_manager = UserConfigManager()

        print(f"[Target] Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if compiled_model_path and os.path.isfile(compiled_model_path):
            print(f"[Target] Loading precompiled model from {compiled_model_path}")
            self.model = torch.jit.load(compiled_model_path)
        else:
            print(f"[Target] Loading HF model {model_path} in BF16 for Trainium.")
            base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
            base_model.eval()

            if compile_model:
                print("[Target] Compiling model with torch_neuronx.trace...")
                # Build the wrapper that returns (logits, pkv) as a 2-tuple
                wrapper = LlamaKVWrapper(base_model)

                num_layers = base_model.config.num_hidden_layers
                num_kv = getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)
                head_dim = getattr(base_model.config, "head_dim", base_model.config.hidden_size // num_kv)
                batch_size = 1
                seq_len   = 1
                past_seq  = max_context - 1

                dummy_past = []
                for _ in range(num_layers):
                    k = torch.zeros(batch_size, num_kv, past_seq, head_dim, dtype=torch.bfloat16)
                    v = torch.zeros(batch_size, num_kv, past_seq, head_dim, dtype=torch.bfloat16)
                    dummy_past.append((k, v))
                dummy_past = tuple(dummy_past)

                dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

                # Now we compile
                self.model = torch_neuronx.trace(
                    wrapper,
                    example_inputs=(dummy_input_ids, dummy_past)
                )
                if compiled_model_path:
                    print(f"[Target] Saving compiled model to {compiled_model_path}")
                    self.model.save(compiled_model_path)
            else:
                # CPU fallback
                self.model = base_model

        # Optionally move trace to device
        try:
            torch_neuronx.experimental.set_neuron_cores(1)
            torch_neuronx.move_trace_to_device(self.model, 0)
        except Exception as e:
            print(f"[Target] Could not move model to Neuron device: {e}")

        print("[Target] ModelServiceServicer init complete.")

    def PrepareSpeculative(self, request, context):
        # ...
        return model_service_pb2.PrepareSpeculativeResponse(first_tokens="Not implemented")

    def VerifyTokens(self, request, context):
        # ...
        return model_service_pb2.VerifyTokensResponse(verified=True)

    def GenerateContent(self, request, context):
        # ...
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