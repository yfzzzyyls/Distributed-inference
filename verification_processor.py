import redis
import torch
import json
import time
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import parse_arguments, rollback_past_key_values

class VerificationProcessor:
    def __init__(self, args):
        # Initialize distributed accelerator
        self.accelerator = Accelerator()
        self.args = args
        
        # Load model and tokenizer to the assigned device
        self.load_model()
        # Connect to Redis
        self.redis = redis.Redis(host="localhost", port=6379, db=0)

        # Set queue names based on process ID
        self.process_id = self.accelerator.process_index
        self.task_queue = f"task_queue_{self.process_id}"

        # Initialize cache for past_key_values by request ID
        self.past_key_values_cache = {}
    
    def load_model(self):
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

    def encode_input(self, text=None, last_token_ids=None):
        print(text, last_token_ids)
        if text is None and last_token_ids is None:
            raise ValueError("Either text or last_token_ids must be provided.")
        if text is not None and last_token_ids is not None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            print(f"draft input_ids shape: {input_ids.shape}")
            input_ids = torch.cat((last_token_ids, input_ids), dim=-1)
        elif text is not None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        elif last_token_ids is not None:
            input_ids = last_token_ids
        return input_ids, len(input_ids[0]) - 1

    def process_next(self, request):
        """Initialize a request by processing the initial token generation."""
        req_id = request["request_id"]
        text = request.get("text", None)

        # Encode input
        past_key_values, last_token_ids = self.past_key_values_cache.get(req_id, (None, None))
        input_ids, _ = self.encode_input(text, last_token_ids)
        # Run model inference without past key values for initialization
        with torch.no_grad():
            Mp = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        # Get the next token and generate the initial verified text
        next_token_logits = Mp.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generate_finished = False if next_token_id[0].item() != self.tokenizer.eos_token_id else True

        next_text = ""
        if next_token_id in self.tokenizer.all_special_ids:
            if next_token_id == self.tokenizer.eos_token_id:
                generate_finished = True
            else:
                next_text = self.tokenizer.decode(next_token_id)
                print(f"next special token: {next_text}")
        else:
            next_text = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
            if next_text == "":
                next_text = self.tokenizer.decode(next_token_id)
                print(f"empty next text: {next_token_id}, {next_text}")

        # Cache the past_key_values
        self.past_key_values_cache[req_id] = (Mp.past_key_values, next_token_id.unsqueeze(0))
        # output = torch.cat((input_ids[0], next_token_id), dim=0)
        return {"request_id": req_id, "next_text": next_text, "generate_finished": generate_finished}

    def process_verify(self, request):
        """Verify continuation of text using cached past_key_values."""
        req_id = request["request_id"]
        text = request["text"]
        # gamma = request.get("gamma", 1)

        # Retrieve past_key_values from the cache
        past_key_values, last_token_ids = self.past_key_values_cache.get(req_id)
        print(f"last_token_ids shape: {last_token_ids.shape}")
        # generate_step = int(gamma) + 1

        # Encode input with the past key values
        draft_input_ids, draft_length = self.encode_input(text, last_token_ids)
        generate_step = draft_length
        print(f"Total Draft input IDs shape: {draft_input_ids.shape}")
        print(f"last past_key_values shape: {past_key_values[0][0].shape}")
        # Run model inference with past key values
        with torch.no_grad():
            Mp = self.model(input_ids=draft_input_ids, past_key_values=past_key_values, use_cache=True)

        # Update cache with new past key values
        target_logits = Mp.logits
        # print(f"Target logits shape: {target_logits.shape}")
        print(f"target logits shape: {target_logits.shape}")

        # Verify the generated tokens
        predicted_token_ids = target_logits[0, :, :].argmax(dim=-1)
        # print(f"Predicted token IDs shape: {predicted_token_ids.shape}")
        n = generate_step
        for j in range(generate_step):
            if predicted_token_ids[j] == draft_input_ids[0][j+1]:
                continue
            else:
                n = j
                break
        print(f"passed draft n: {n}")
        # Generate verified text based on verification results
        verified_ids = torch.cat((draft_input_ids[0, :n], predicted_token_ids[n:n+1]), dim=-1)
        generate_finished = False if predicted_token_ids[n] != self.tokenizer.eos_token_id else True
        print(f"verified ids shape: {verified_ids.shape}")

        if not generate_finished:
            rollbacked_past_key_values = rollback_past_key_values(Mp.past_key_values, generate_step - n)
            self.past_key_values_cache[req_id] = (rollbacked_past_key_values, predicted_token_ids[n].unsqueeze(0).unsqueeze(0))
        else:
            self.past_key_values_cache.pop(req_id)
        print(f"deal with past_key_values shape: {rollbacked_past_key_values[0][0].shape}")
        next_text = ""
        if verified_ids[-1] in self.tokenizer.all_special_ids:
            if verified_ids[-1] == self.tokenizer.eos_token_id:
                generate_finished = True
            else:
                next_text = self.tokenizer.decode(verified_ids[-1])
                print(f"next special token: {next_text}")
        else:
            next_text = self.tokenizer.decode(verified_ids[-1], skip_special_tokens=True)

            if next_text == "":
                next_text = self.tokenizer.decode(verified_ids[-1])
                print(f"empty next text: {verified_ids[-1]}, {next_text}")

        return {"request_id": req_id, "next_text": next_text, "passed_tokens": n, "correct_rate": n / generate_step, "generate_finished": generate_finished}

    def process_task(self, request):
        """Select the correct process method based on request type."""
        job_id = request["request_id"]
        job_type = request["request_type"]
        result_queue = f"result_queue_{job_id}"
        if job_type == "init":
            print(f"Process {self.process_id} initializing task: {request}")
            return (result_queue, self.process_next(request))
        elif job_type == "verify":
            print(f"Process {self.process_id} verifying task: {request}")
            return (result_queue, self.process_verify(request))
        elif job_type == "delete":
            print(f"Process {self.process_id} deleting task: {request}")
            self.past_key_values_cache.pop(job_id, None)
            torch.cuda.empty_cache()
            return (result_queue, {"request_id": job_id, "status": "success"})
        else:
            raise ValueError(f"Unknown task type: {job_type}")

    def listen_for_tasks(self):
        """Continuously listen for tasks from Redis and process them."""
        # print(f"Process {self.process_id} listening for tasks on {self.task_queue}...")
        while True:
            try:
                # Blocking pop from Redis to fetch a task
                task = self.redis.blpop(self.task_queue, timeout=10)
                if task:
                    # Parse task data
                    task_data = json.loads(task[1])
                    # print(f"Process {self.process_id} received task: {task_data}")

                    # Process the task using the selected method
                    result_queue, result = self.process_task(task_data)

                    # Push the result back to Redis result queue
                    self.redis.rpush(result_queue, json.dumps(result))
                    print(f"Process {self.process_id} completed task: {result}")
                else:
                    time.sleep(0.1)  # Pause briefly if no task is received
                    # print(f"Process {self.process_id} no task received, continuing to listen...")
            except Exception as e:
                print(f"Process {self.process_id} encountered error: {e}")
                error_message = {
                    "status": "error",
                    "message": str(e),
                    "task_id": task_data.get("task_id", "unknown") if 'task_data' in locals() else "unknown"
                }
                result_queue = task_data.get("result_queue", "default_result_queue") if 'task_data' in locals() else "default_result_queue"
                # Push the error message back to the Redis result queue
                self.redis.rpush(result_queue, json.dumps(error_message))
                print(f"Process {self.process_id} encountered error: {e}")
                time.sleep(1)  # Pause briefly if an error occurs
                time.sleep(1)  # Pause briefly if an error occurs

def main(args):
    # Initialize VerificationProcessor and start listening for tasks
    processor = VerificationProcessor(args)
    processor.listen_for_tasks()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
