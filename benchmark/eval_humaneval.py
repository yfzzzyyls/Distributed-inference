import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import ipdb
import random
import concurrent.futures
import numpy as np
from collections import Counter
from batch_client import BatchClient

class EvalHumaneval():
    def __init__(self, client_num=4, host='localhost', port=50051, model_path="/home/apc/llama/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        self.client_num = client_num

        # load relative resources
        self.load_data()
        self.init_client(host, port, model_path)
        
        self.draft_time = []
        self.target_time = []
        self.acc_num = []
    
    def init_client(self, host, port, model_path):
        self.clients = []
        for i in range(self.client_num):
            client = BatchClient(host, port, model_path)
            self.clients.append(client)

    def load_data(self):
        # * load evaluation data
        print(f"Loading HumanEval data...", 3)
        data = []
        with open(os.path.join("/home/apc/NYU/SaiLab/Distributed_SD/MultiDeviceSpeculativeDecoding/data", "humaneval.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                #encode_special_token_flag = not ("Llama-3.1" in self.args.draft_model and "Llama-3.1" in self.args.target_model)
                #input_ids = self.tokenizer.encode(datum["input_text"], add_special_tokens=False)
                #datum["input_ids"] = torch.tensor(input_ids).unsqueeze(0)
                data.append(datum)
        
        self.data = np.array_split(data, self.client_num)
    
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
    def client_decode(self, client, data):  
        print(f"Start evaluating...")
        out_path = os.path.join("/home/apc/NYU/SaiLab/Distributed_SD/MultiDeviceSpeculativeDecoding/result", f"Qwen_humaneval.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time":[], "num_tokens":[]}

        for datum in tqdm.tqdm(data, total=len(data), ncols=50):
            input_text = datum["input_text"]
            #input_ids = datum["input_ids"]
            start_time = time.time()
            generate_ids, timestamps = client.speculative_decoding(input_text, 40)
            end_time = time.time()
            if datum["task_id"] != "HumanEval/0":
                # skip the first prompt time consumption
                wall_times["time"].append(end_time-start_time)
                wall_times["num_tokens"].append(40)
            output = generate_ids
            out_f.write(json.dumps({"task_id": datum["task_id"], "time": end_time-start_time, "new_tokens": 40, "completion": output}, ensure_ascii=False) + "\n")
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
    
    def eval(self):
        start = time.time()
        #print(self.clients, self.data)
        print(f"Start evaluating...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.client_num) as executor:
            futures = [
                        executor.submit(self.client_decode, self.clients[i], self.data[i]) 
                        for i in range(self.client_num)  
                    ]
        concurrent.futures.wait(futures)

        print(f"Total time: {time.time()-start}")
        

if __name__ == "__main__":
    alg = EvalHumaneval(1)
    alg.eval()