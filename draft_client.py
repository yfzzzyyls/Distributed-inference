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

class LlamaKVWrapper(torch.nn.Module):
    def __init__(self, base_model, num_layers):
        super().__init__()
        self.base_model = base_model
        self.num_layers = num_layers

    def forward(self, input_ids, past_key_values=None):
        # Re-chunk your past_key_values if it's flattened
        if past_key_values is not None:
            if len(past_key_values) == 2 * self.num_layers:
                chunked = []
                for i in range(0, len(past_key_values), 2):
                    k = past_key_values[i]
                    v = past_key_values[i + 1]
                    chunked.append((k, v))
                past_key_values = tuple(chunked)
            else:
                raise ValueError(
                    f"Got {len(past_key_values)} items, expected {2 * self.num_layers}."
                )

        # Call the underlying Hugging Face LLaMA
        outputs = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Instead of returning a dict, return (Tensor, Tuple[...])
        # This is a stable 2-tuple for the tracer: (logits, nested-tuple-of-tensors)
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

        # Connect to the server
        server_addr = f"{host}:{port}"
        self.channel = grpc.insecure_channel(server_addr)
        self.stub = model_service_pb2_grpc.ModelServiceStub(self.channel)

        # Load or compile draft model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # If precompiled:
        if compiled_model_path and os.path.isfile(compiled_model_path):
            print(f"[Draft] Loading precompiled model from {compiled_model_path}")
            self.model = torch.jit.load(compiled_model_path)
        else:
            print(f"[Draft] Loading HF model {model_name} in BF16 for Trainium.")
            base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            base_model.eval()

            if compile_model:
                print("[Draft] Compiling draft model with torch_neuronx.trace…")
                seq_len = 1
                past_seq = max_length - 1

                num_layers = base_model.config.num_hidden_layers
                # Use num_key_value_heads for K/V dimension count
                num_kv = getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)
                # IMPORTANT: use `head_dim` from the config directly
                head_dim = base_model.config.head_dim
                batch_size = 1

                dummy_past_pairs = []
                for _ in range(num_layers):
                    k = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    v = torch.zeros((batch_size, num_kv, past_seq, head_dim), dtype=torch.bfloat16)
                    dummy_past_pairs.append((k, v))

                # Flatten each layer's (k, v) into a single tuple for legacy cache
                flattened = []
                for (k, v) in dummy_past_pairs:
                    flattened.append(k)
                    flattened.append(v)
                dummy_past = tuple(flattened)

                dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

                wrapper = LlamaKVWrapper(base_model, num_layers)
                self.model = torch_neuronx.trace(wrapper, (dummy_input_ids, dummy_past))

                if compiled_model_path:
                    print(f"[Draft] Saving compiled model to {compiled_model_path}")
                    self.model.save(compiled_model_path)
            else:
                # No compile -> run on CPU
                self.model = base_model

        # Attempt to move the compiled model to device
        try:
            torch_neuronx.experimental.set_neuron_cores(1)
            torch_neuronx.move_trace_to_device(self.model, 0)
        except Exception as e:
            print(f"[Draft] Could not move draft model to Neuron device: {e}")

    def speculative_decode(self):
        """
        Full speculation: we call the server to get the first token, then we generate gamma tokens at a time.
        """
        # 1) Prepare a gRPC request to server to set up the session
        user_uuid = "draft-user-1234"  # just a static example
        prepare_req = model_service_pb2.PrepareSpeculativeRequest(
            user_uuid=user_uuid,
            prompt=self.prompt,
            max_length=self.max_length,
            generate_step=self.gamma,
            exact_mode=True,
            debug_mode=False
        )
        prepare_resp = self.stub.PrepareSpeculative(prepare_req)
        # The server returns prompt + first token
        first_tokens_text = prepare_resp.first_tokens
        # We'll feed that text into the draft model to build up the KV cache
        context_ids = self.tokenizer(first_tokens_text, return_tensors="pt").input_ids
        # Warm up the draft model
        local_past = None
        for tid in context_ids[0]:
            with torch.no_grad():
                # out = self.model(tid.view(1,1), local_past)
                out = self.model(input_ids=tid.view(1,1), past_key_values=local_past)
            # local_past = out.past_key_values
            logits, local_past = out

        generated_text = first_tokens_text
        tokens_generated = len(self.tokenizer.tokenize(first_tokens_text))

        # Keep generating in chunks of gamma
        while tokens_generated < self.max_length:
            chunk_tokens = []
            for step_i in range(self.gamma):
                if tokens_generated >= self.max_length:
                    break
                with torch.no_grad():
                    # out = self.model(torch.tensor([[0]]), local_past)  # or last token? 
                    out = self.model(input_ids=torch.tensor([[0]]),past_key_values=local_past)
                    # Actually we need the last token from the prior step if we had it
                    # Simplify by always using a dummy? This is incomplete, see real logic below:
                    logits, local_past = out
                    logits = out.logits[..., -1, :]
                    # local_past = out.past_key_values
                    next_id = int(torch.argmax(logits, dim=-1))
                    chunk_tokens.append(next_id)
                    generated_text += self.tokenizer.decode(next_id, skip_special_tokens=True)
                    tokens_generated += 1

            # Send chunk_tokens to server for verification
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            verify_req = model_service_pb2.VerifyTokensRequest(
                user_uuid=user_uuid,
                token_text=chunk_text
            )
            verify_resp = self.stub.VerifyTokens(verify_req)
            if not verify_resp.verified:
                print("[Draft] Server rejected tokens. (Simplified logic -> stop or fallback.)")
                break

        return generated_text

if __name__=="__main__":
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

    # If we're just compiling, skip the gRPC call to server
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