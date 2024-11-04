import argparse
import os

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')
    
    parser.add_argument('--data_path', type=str, default="./data")

    parser.add_argument('--draft_model', type=str, default="/home/apc/llama/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--target_model', type=str, default="/home/apc/llama/Llama-3.2-3B-Instruct")
    
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=str, default="50051")
    parser.add_argument('--use_cache', type=bool, default=True)

    parser.add_argument('--exp_name', '-e', type=str, default="./exp", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="small", choices=["small", "large", "sd", "para_sd", "para_sd_wo_1", "para_sd_wo_2"], help='eval mode.')
    parser.add_argument('--batch_size', '-n', type=int, default=2, help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=128, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    return args