# utils.py
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Distributed inference on AWS Trainium")

    # Common arguments for draft & target
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname/IP for gRPC")
    parser.add_argument("--port", type=str, default="50051", help="Server port for gRPC")
    
    # Draft model arguments
    parser.add_argument("--draft_model", type=str, default="/home/ubuntu/models/llama-3.2-1b",
                        help="Path/ID for the LLaMA 3.2 1B draft model (HF format)")
    parser.add_argument("--compiled_draft_model", type=str, default=None,
                        help="Path to a precompiled draft model TorchScript (if available)")
    
    # Target model arguments
    parser.add_argument("--target_model", type=str, default="/home/ubuntu/models/llama-3.2-3b",
                        help="Path/ID for the LLaMA 3.2 3B target model (HF format)")
    parser.add_argument("--compiled_target_model", type=str, default=None,
                        help="Path to a precompiled target model TorchScript (if available)")

    # Speculative decoding parameters
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="Use KV cache for incremental decoding (enabled by default).")
    parser.add_argument("--gamma", type=int, default=4,
                        help="Number of tokens to generate speculatively at once (draft model).")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max tokens to generate in total for the prompt.")
    
    # Neuron compile flags
    parser.add_argument("--compile", action="store_true",
                        help="If set, compile the model on-the-fly with torch_neuronx.trace")
    parser.add_argument("--max_context", type=int, default=2048,
                        help="Max context length for compilation (KV cache).")

    # Additional flags for server or client
    parser.add_argument("--server", action="store_true",
                        help="If set, we run the target model server (model_service.py).")
    parser.add_argument("--draft", action="store_true",
                        help="If set, we run the draft client (draft_client.py).")
    parser.add_argument("--quantize", action="store_true",
                        help="(Not used) For int4 or int8 quant; ignoring for Neuron BF16.")
    
    # Evaluate / test
    parser.add_argument("--prompt", type=str, default="Once upon a time,",
                        help="Prompt to use if we test or evaluate.")
    
    return parser.parse_args()