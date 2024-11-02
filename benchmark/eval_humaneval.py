import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
from utils import parse_arguments
import ipdb
import random
import concurrent.futures
import numpy as np
from collections import Counter
from batch_client import BatchClient
import argparse


class EvalHumaneval(BatchClient):
    def __init__(self, args):
        super().__init__(args)
        self.client_num = self.accelerator.num_processes

        # load relative resources
        self.load_model()
        self.load_data()
        
        self.draft_time = []
        self.target_time = []
        self.acc_num = []

    def load_data(self):
        # * load evaluation data
        print(f"Loading HumanEval data...", 3)     
        num_processes = self.accelerator.num_processes
        process_id = self.accelerator.process_index

        # 统计文件的总行数
        file_path = os.path.join(self.args.data_path, "humaneval.jsonl")
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        # 计算每个进程应读取的行范围
        lines_per_process = total_lines // num_processes
        start_line = process_id * lines_per_process
        end_line = start_line + lines_per_process if process_id < num_processes - 1 else total_lines

        # 读取文件中对应行数的分片
        data = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                elif i >= end_line:
                    break
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                data.append(datum)
        
        self.data = data
        print(f"Process {process_id} loaded {len(data)} items from lines {start_line} to {end_line}.")
    
    def preprocess(self, input_text):
        text = input_text.strip()
        return text

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:] # tokenizer will add a '<s> ' at the beginning of the text.
        else:
            generation = output_text[len(input_text):]
        stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                next_line = generation.index(stop_word)
                generation = generation[:next_line].strip()
        output_text = input_text + '\n    ' + generation
        output_text = output_text.replace("\t", "    ")
        
        return output_text
             
    @torch.no_grad()
    def eval(self):  
        print(f"Start evaluating...")
        out_path = os.path.join("/home/apc/NYU/SaiLab/Distributed_SD/MultiDeviceSpeculativeDecoding/result", f"Qwen_humaneval.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time":[], "num_tokens":[]}

        for datum in tqdm.tqdm(self.data, total=len(self.data), ncols=50):
            input_text = datum["input_text"]
            #input_ids = datum["input_ids"]
            start_time = time.time()
            generate_ids, timestamps = self.speculative_decoding(input_text)
            end_time = time.time()
            if datum["task_id"] != "HumanEval/0":
                # skip the first prompt time consumption
                wall_times["time"].append(end_time-start_time)
                wall_times["num_tokens"].append(self.args.max_tokens)
            output = generate_ids
            out_f.write(json.dumps({"task_id": datum["task_id"], "time": end_time-start_time, "new_tokens":  self.args.max_tokens, "completion": output}, ensure_ascii=False) + "\n")
            out_f.flush()
        
        out_f.close()
        
        #self.color_print(f"current eval mode: {self.args.eval_mode}", 0)
        #self.color_print(f"draft model forward times: {self.draft_forward_times}", 2)
        
        #self.accelerator.wait_for_everyone()
        #
        #if (self.accelerator.num_processes == 1 and self.accelerator.is_main_process) or (self.accelerator.num_processes == 2 and not self.accelerator.is_main_process):
        #    print(f"\033[92mtarget model forward times: {self.target_forward_times}\033[0m")
        #
        #self.accelerator.wait_for_everyone()
        #
        #if self.accelerator.is_main_process:
        #    speed = sum(wall_times["num_tokens"]) / sum(wall_times["time"])
        #    speed_std = (torch.tensor(wall_times["num_tokens"]) / torch.tensor(wall_times["time"])).std().item()
        #    self.color_print(f"generate speed (tokens / second):  {speed:.2f} with std {speed_std}", 2)
#
        #if self.accelerator.is_main_process:
        #    try:
        #        self.color_print(f"Mean accepted tokens: {sum(self.num_acc_tokens) / len(self.num_acc_tokens)}")
        #    except:
        #        pass

if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalHumaneval(args)
    alg.eval()