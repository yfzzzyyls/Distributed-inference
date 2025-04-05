# evaluate_test.py
import time
import argparse
import grpc
import torch
from protos import model_service_pb2, model_service_pb2_grpc
from draft_client import ModelServiceClient


def main():
    parser = argparse.ArgumentParser(description="Evaluate speculative decoding performance")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="50051")
    parser.add_argument("--draft_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--compiled_draft_model", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--target_model_service", action="store_true",
                        help="If set, we also measure direct target generation via GenerateContent RPC.")
    args = parser.parse_args()

    # 1) Speculative decoding
    client = ModelServiceClient(
        model_name=args.draft_model,
        compiled_model_path=args.compiled_draft_model,
        compile_model=args.compile,
        max_length=args.max_tokens,
        gamma=args.gamma,
        host=args.host,
        port=args.port,
        prompt=args.prompt
    )
    st = time.time()
    spec_output = client.speculative_decode()
    et = time.time()
    spec_elapsed = et - st
    spec_tokens = len(spec_output.split())
    print(f"\nSpeculative decoding result:\n{spec_output}")
    print(f"Spec time: {spec_elapsed:.2f}s, tokens={spec_tokens}, throughput={spec_tokens/spec_elapsed:.2f} t/s")

    # 2) If we want to measure target-only
    if args.target_model_service:
        channel = grpc.insecure_channel(f"{args.host}:{args.port}")
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        req = model_service_pb2.GenerateContentRequest(prompt=args.prompt, max_length=args.max_tokens)
        st2 = time.time()
        resp = stub.GenerateContent(req)
        et2 = time.time()
        base_elapsed = et2 - st2
        base_tokens = len(resp.generated_text.split())
        print(f"\nBaseline target-only result:\n{resp.generated_text}")
        print(f"Baseline time: {base_elapsed:.2f}s, tokens={base_tokens}, throughput={base_tokens/base_elapsed:.2f} t/s")


if __name__=="__main__":
    main()