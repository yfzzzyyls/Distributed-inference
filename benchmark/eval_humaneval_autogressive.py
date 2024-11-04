import time
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch
import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
from utils import parse_arguments
import os
import tqdm
import json

# 全局变量
request_queue = []
request_text = {}
request_past_key_values = {}

results = {}
BATCH_SIZE = 2
MAX_WAIT_TIME = 1
generate_step = 4


# 实现服务
class VerficationServer():
    def __init__(self, args):
        self.args = args
        self.load_model()
        self.load_data()

    def load_model(self):
        quantization_config = QuantoConfig(weights="int4")
        self.model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.float16)  # 替换成实际的模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.target_model)
        self.device = self.model.device

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

    def load_data(self):
        # * load evaluation data
        print(f"Loading HumanEval data...")
        data = []
        total_lines = 100
        with open(os.path.join(self.args.data_path, "humaneval.jsonl")) as f:
            for i, line in enumerate(f):
                if i >= total_lines:
                    break
                datum = json.loads(line)
                datum["input_text"] = self.preprocess(datum["prompt"])
                data.append(datum)
        self.data = data

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
        out_path = os.path.join(self.args.exp_name, f"Qwen_humaneval_autogressive.jsonl")
        out_f = open(out_path, "a")
        wall_times = {"time":[], "num_tokens":[]}

        for datum in tqdm.tqdm(self.data, total=len(self.data), ncols=50):
            input_text = datum["input_text"]
            #input_ids = datum["input_ids"]
            start_time = time.time()
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            past_key_values = None
            generated_tokens = []
            for _ in range(self.args.max_tokens):
                with torch.no_grad():
                    output = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                logits = output.logits
                past_key_values = output.past_key_values

                # Get the next token
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Shape: [batch_size, 1]

                # Append the generated token
                generated_tokens.append(next_token_id.item())

                # Update input_ids for the next iteration
                input_ids = next_token_id  # Shape: [batch_size, 1]

                # Check for end-of-sequence token
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
            end_time = time.time()
            if datum["task_id"] != "HumanEval/0":
                # skip the first prompt time consumption
                wall_times["time"].append(end_time-start_time)
                wall_times["num_tokens"].append(self.args.max_tokens)
            output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            out_f.write(json.dumps({"task_id": datum["task_id"], "time": end_time-start_time, "new_tokens":  self.args.max_tokens, "completion": output}, ensure_ascii=False) + "\n")
            out_f.flush()
        
        out_f.close()
        print(f"Finish evaluating..., Autogressive eval total time: {sum(wall_times['time'])}")

             
if __name__ == '__main__':
    args = parse_arguments()
    alg = VerficationServer(args)
    alg.eval()
