# draft_client.py
import os
import time
import grpc
import torch
import torch_neuronx
import gevent
import gevent.pool

from transformers import AutoModelForCausalLM, AutoTokenizer
from protos import model_service_pb2, model_service_pb2_grpc

def create_empty_past(num_layers, num_kv, past_seq, head_dim):
    """
    Build a nested tuple: ((k1,v1),(k2,v2),...).
    Each (k,v) is [1, num_kv, past_seq, head_dim].
    """
    dummy_past_list = []
    for _ in range(num_layers):
        k = torch.zeros((1, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
        v = torch.zeros((1, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
        dummy_past_list.append((k, v))
    return tuple(dummy_past_list)

class LlamaKVWrapper(torch.nn.Module):
    def __init__(self, base_model, num_layers):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers

    def forward(self, input_ids, past_key_values=None):
        if past_key_values is not None and len(past_key_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} (k,v) pairs, got {len(past_key_values)}"
            )
        outputs = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        return (outputs.logits, outputs.past_key_values)

class ModelServiceClient:
    def __init__(self,
                 model_name: str,
                 compiled_model_path: str=None,
                 compile_model: bool=False,
                 max_length: int=128,
                 gamma: int=4,
                 host: str="localhost",
                 port: str="50051",
                 prompt: str="Once upon a time,"):
        self.host = host
        self.port = port
        self.max_length = max_length
        self.gamma = gamma
        self.prompt = prompt

        server_addr = f"{host}:{port}"
        self.channel = grpc.insecure_channel(server_addr)
        self.stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model = None
        base_config = None

        if compiled_model_path and os.path.isfile(compiled_model_path):
            print(f"[Draft] Loading precompiled model from {compiled_model_path}")
            self.model = torch.jit.load(compiled_model_path)
        else:
            print(f"[Draft] Loading HF model {model_name} in BF16 for Trainium.")
            base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            base_model.eval()
            base_config = base_model.config

            if compile_model:
                print("[Draft] Compiling draft model with torch_neuronx.trace…")

                # We'll compile so the draft can handle [1,32]
                seq_len = 32
                past_seq = max_length - 1

                num_layers = base_config.num_hidden_layers
                num_kv = getattr(base_config, "num_key_value_heads", base_config.num_attention_heads)
                head_dim = base_config.head_dim
                batch_size = 1

                # Nested dummy past
                dummy_past_list = []
                for _ in range(num_layers):
                    k = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    v = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    dummy_past_list.append((k, v))
                dummy_past = tuple(dummy_past_list)

                # [1,32] input_ids
                dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

                wrapper = LlamaKVWrapper(base_model, num_layers)
                self.model = torch_neuronx.trace(wrapper, (dummy_input_ids, dummy_past))

                if compiled_model_path:
                    print(f"[Draft] Saving compiled model to {compiled_model_path}")
                    self.model.save(compiled_model_path)
            else:
                self.model = base_model

        if base_model is not None:
            base_config = base_model.config

        if base_config is not None:
            num_layers = base_config.num_hidden_layers
            num_kv = getattr(base_config, "num_key_value_heads", base_config.num_attention_heads)
            head_dim = base_config.head_dim
            past_seq = self.max_length - 1
        else:
            print("[Draft] No base_config found; using fallback shapes.")
            num_layers = 16
            num_kv = 8
            head_dim = 64
            past_seq = self.max_length - 1

        # Build a nested empty past
        self.empty_past = create_empty_past(num_layers, num_kv, past_seq, head_dim)

        # Not pinning to single core
        print("[Draft] ModelServiceClient init complete.")

    def speculative_decode(self):
        user_uuid = "draft-user-1234"
        prepare_req = model_service_pb2.PrepareSpeculativeRequest(
            user_uuid=user_uuid,
            prompt=self.prompt,
            max_length=self.max_length,
            generate_step=self.gamma,
            exact_mode=True,
            debug_mode=False
        )
        prepare_resp = self.stub.PrepareSpeculative(prepare_req)
        first_tokens_text = prepare_resp.first_tokens

        # 1) Convert to input_ids => shape [1, N]
        context_ids = self.tokenizer(first_tokens_text, return_tensors="pt").input_ids
        
        # 2) Pad up to [1,32] if needed
        real_len = context_ids.size(1)
        if real_len < 32:
            pad_len = 32 - real_len
            # If your model has a real pad token, use that. We use 0 here:
            pad_tensor = torch.zeros((1, pad_len), dtype=torch.long)
            context_ids = torch.cat([context_ids, pad_tensor], dim=1)  # shape [1,32]
        elif real_len > 32:
            raise ValueError(f"Prompt too long ({real_len} tokens); compile_time=32")

        local_past = self.empty_past

        # Single pass for all prompt tokens => shape [1,32]
        with torch.no_grad():
            logits, local_past = self.model(context_ids, local_past)

        generated_text = first_tokens_text
        tokens_generated = len(self.tokenizer.tokenize(first_tokens_text))

        while tokens_generated < self.max_length:
            chunk_tokens = []
            for _ in range(self.gamma):
                if tokens_generated >= self.max_length:
                    break
                with torch.no_grad():
                    # Single-token pass => shape [1,1]
                    single_ids = torch.zeros((1,1), dtype=torch.long)
                    logits, local_past = self.model(single_ids, local_past)
                    next_id = int(torch.argmax(logits[..., -1, :], dim=-1))
                    chunk_tokens.append(next_id)
                    generated_text += self.tokenizer.decode(next_id, skip_special_tokens=True)
                    tokens_generated += 1

            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            verify_req = model_service_pb2.VerifyTokensRequest(
                user_uuid=user_uuid,
                token_text=chunk_text
            )
            verify_resp = self.stub.VerifyTokens(verify_req)
            if not verify_resp.verified:
                print("[Draft] Server rejected tokens.")
                break

        return generated_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Draft Model Client on AWS Trainium")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="50051")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--compiled_model_path", type=str, default=None)
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    args = parser.parse_args()

    client = ModelServiceClient(
        model_name=args.model,
        compiled_model_path=args.compiled_model_path,
        compile_model=args.compile_model,
        max_length=args.max_length,
        gamma=args.gamma,
        host=args.host,
        port=args.port,
        prompt=args.prompt
    )

    if args.compile_model:
        print("[Draft] Model compiled and saved. Exiting.")
        import sys
        sys.exit(0)

    start_t = time.time()
    output = client.speculative_decode()
    end_t = time.time()
    elapsed = end_t - start_t
    tokens_out = len(output.split())
    print(f"\n[Draft] Final output: {output}")
    print(f"[Draft] Time: {elapsed:.3f} s, tokens: {tokens_out}, speed: {tokens_out/elapsed:.2f} tokens/s")