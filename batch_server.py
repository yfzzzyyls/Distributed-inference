import time
import threading
import grpc
from concurrent import futures
import protos.batch_pb2 as batch_pb2
import protos.batch_pb2_grpc as batch_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch
from utils import parse_arguments
import traceback


# 全局变量
request_queue = []
request_text = {}
request_past_key_values = {}

results = {}
condition = threading.Condition()
queue_lock = threading.Lock()
BATCH_SIZE = 2
MAX_WAIT_TIME = 1
generate_step = 4


# 实现服务
class VerficationServer(batch_pb2_grpc.BatchServiceServicer):
    def __init__(self, args):
        self.args = args
        self.load_model()
        # 启动批处理线程
        self.batch_thread = threading.Thread(target=self.process_batch_parallel_with_chunk, daemon=True)
        self.batch_thread.start()
        self.monitor_thread = threading.Thread(target=self.monitor_batch_thread, daemon=True)
        self.monitor_thread.start()

    def monitor_batch_thread(self):
        """Monitors the batch_thread to check if it's alive."""
        while True:
            time.sleep(2)  # Check every 2 seconds
            if not self.batch_thread.is_alive():
                print("Batch thread has stopped. Printing stack trace:")
                stack_trace = traceback.format_stack()
                print("".join(stack_trace))
                # Optionally, restart the batch thread
                #self.batch_thread = threading.Thread(target=self.process_batch_parallel, daemon=True)
                self.batch_thread.start()
                print("Restarted batch thread.")

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


    def AddRequest(self, request, context):
        """处理 AddRequest 请求并等待批处理完成后返回"""
        global request_queue, request_text, results
        with queue_lock:  # 使用锁保护共享资源
            if request.request_type == "init":
                request_queue.append({"request_id": request.request_id, "text": request.text, "request_type": "init", "gamma": generate_step})
                request_text[request.request_id] = request.text
                print(f"Initialized request {request.request_id}: {request.text}")

                # if request.request_id not in request_text:
                #     request_queue.append({"request_id": request.request_id, "text": request.text, "request_type": "init", "gamma": generate_step})
                #     request_text[request.request_id] = request.text
                #     print(f"Initialized request {request.request_id}: {request.text}")
                # else:
                #     print(f"Request {request.request_id} already initialized.")
            elif request.request_type == "update":
                if request.request_id in request_text:
                    request_text[request.request_id] += request.text
                    print(f"Updated request {request.request_id}: {request_text[request.request_id]}")
                    return batch_pb2.ResultResponse(
                        request_id=request.request_id,
                        verified_text="update success",
                    )
                else:
                    print(f"Request {request.request_id} not found. Cannot update.")
                    return batch_pb2.ResultResponse(
                        request_id=request.request_id,
                        verified_text="update failed",
                    )
            elif request.request_type == "verify":
                if request.request_id not in request_queue:
                    if request.text != "":
                        request_queue.append({"request_id": request.request_id, "text": request_text[request.request_id], "request_type": "verify", "gamma": int(request.text)})
                    else:
                        request_queue.append({"request_id": request.request_id, "text": request_text[request.request_id], "request_type": "verify", "gamma": generate_step})
                    print(f"Verification request {request.request_id} added to batch queue.")
            elif request.request_type == "delete":
                if request.request_id in request_text:
                    del request_text[request.request_id]
                    del request_past_key_values[request.request_id]
                    print(f"Deleted request {request.request_id}.")
                    return batch_pb2.ResultResponse(
                        request_id=request.request_id,
                        verified_text="delete success",
                    )
                else:
                    print(f"Request {request.request_id} not found. Cannot delete.")
                    return batch_pb2.ResultResponse(
                        request_id=request.request_id,
                        verified_text="delete failed",
                    )

        # 等待批处理完成并获取结果
        with condition:
            condition.wait_for(lambda: request.request_id in results)
            with queue_lock:  # 访问 results 时加锁
                verified_text = results.pop(request.request_id)
            return batch_pb2.ResultResponse(
                request_id=request.request_id,
                verified_text=verified_text,
            )

    # 批处理逻辑
    def process_batch(self):
        """后台线程，用于批量处理请求"""
        global request_queue, request_text, results
        while True:
            start_time = time.time()  # 每次进入批量处理逻辑时重置 start_time
            batch = []

            # 从请求队列中收集请求，直到达到批次大小或超时
            while len(batch) < self.args.batch_size and (time.time() - start_time) < MAX_WAIT_TIME:
                if request_queue:
                    request = request_queue.pop(0)
                    req_id = request["request_id"]
                    job_type = request["request_type"]
                    job_text = request["text"]
                    gamma = request["gamma"]
                    batch.append((req_id, job_type, job_text, gamma))
                else:
                    time.sleep(0.1)  # 无请求时稍作等待

            if batch:
                # 批量处理请求
                batch_size = len(batch)
                texts = [req for _, _, req, _ in batch]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                #print(f"Processing batch: {texts}, {input_ids}")

                with torch.no_grad():
                    Mp = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        ) 
                target_logits = Mp.logits
                # 遍历批次中的每个序列
                verified_text = [None] * batch_size
                for i in range(batch_size):
                    job_type = batch[i][1]
                    if job_type == "init":
                        print(f"init batch id: {i}, input_text: {texts[i]}")
                        next_token_logits = target_logits[i, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        output = torch.cat((input_ids[i], next_token_id), dim=0)

                        # Convert tensor to text
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        verified_text[i] = texts[i] + generated_text
                        print(f"initialized text: {verified_text[i]}")
                    elif job_type == "verify":
                        # 对 `draft_logits` 中的第 i 个序列进行 `argmax`
                        print(f"verify batch id: {i}, input_text: {texts[i]}")
                        generate_step = self.args.gamma
                        predicted_token_ids = target_logits[i][-generate_step-1:, :].argmax(dim=-1)

                        # 解码预测的 token 和对应输入的 token
                        #predicted_text = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
                        #draft_text = self.tokenizer.decode(input_ids[i][-generate_step:], skip_special_tokens=True)
                        n = generate_step
                        #print(f"predicted_text: {predicted_text}, draft_text: {draft_text}")
                        for j in range(generate_step):
                            if predicted_token_ids[j] == input_ids[i][-generate_step+j]:
                                continue
                            else:
                                n = j
                                break
                        if n == generate_step:
                            verified_text[i] = self.tokenizer.decode(input_ids[i], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)
                        else:
                            verified_text[i] = self.tokenizer.decode(input_ids[i][:-generate_step+n], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)

                        # 比较预测和输入文本
                        #print(f"Sample {i+1}")
                        print("Matched length:", n)

                # 存储批处理结果并通知等待的请求
                with condition:
                    for idx, (req_id,_, _, _) in enumerate(batch):
                        results[req_id] = verified_text[idx]  # 将每个请求的结果存入字典
                        request_text[req_id] = verified_text[idx]  # 更新请求文本
                        print(f"Processed result for {req_id}: {results[req_id]}")
                    condition.notify_all()  # 通知所有等待的请求
            else:
                time.sleep(0.1)

    # 批处理逻辑
    def process_batch_parallel(self):
        """后台线程，用于批量处理请求"""
        global request_queue, request_text, results
        while True:
            start_time = time.time()
            batch = []

            with queue_lock:  # 使用锁来保护 request_queue
                while len(batch) < 1 and (time.time() - start_time) < MAX_WAIT_TIME:
                    if request_queue:
                        request = request_queue.pop(0)
                        req_id = request["request_id"]
                        job_type = request["request_type"]
                        job_text = request["text"]
                        gamma = request["gamma"]
                        batch.append((req_id, job_type, job_text, gamma))
                    else:
                        time.sleep(0.1)

            if batch:
                batch_size = len(batch)
                texts = [req for _, _, req, _ in batch]
                input_ids = self.tokenizer.encode(texts[0], return_tensors="pt").to(self.device)
                # input_ids = inputs.input_ids.to(self.device)
                # attention_mask = inputs.attention_mask.to(self.device)

                with torch.no_grad():
                    Mp = self.model(
                            input_ids=input_ids,
                            use_cache=False,
                        ) 
                target_logits = Mp.logits
                verified_text = [None] * batch_size 

                for i in range(batch_size):
                    job_type = batch[i][1]
                    if job_type == "init":
                        print(f"init batch id: {i}, input_text: {texts[i]}")
                        next_token_logits = target_logits[i, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        output = torch.cat((input_ids[i], next_token_id), dim=0)
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        verified_text[i] = generated_text
                        print(f"initialized text: {verified_text[i]}")
                    elif job_type == "verify":
                        generate_step = int(batch[i][3])
                        print(f"verify batch id: {i}, input_text: {texts[i]}, gamma: {generate_step}")
                        predicted_token_ids = target_logits[i][-generate_step-1:, :].argmax(dim=-1)
                        n = generate_step
                        for j in range(generate_step):
                            if predicted_token_ids[j] == input_ids[i][-generate_step+j]:
                                continue
                            else:
                                n = j
                                break
                        if n == generate_step:
                            verified_text[i] = self.tokenizer.decode(input_ids[i], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)
                        else:
                            verified_text[i] = self.tokenizer.decode(input_ids[i][:-generate_step+n], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)
                        print(f"Verified text: {verified_text[i]}")
                        print(f"Matched length: {n}, draft length: {generate_step}, passed rate: {n/generate_step}")

                with condition:
                    with queue_lock:  # 加锁来保护 results
                        for idx, (req_id, _, _, _) in enumerate(batch):
                            results[req_id] = verified_text[idx]
                            request_text[req_id] = verified_text[idx]
                        condition.notify_all()
            else:
                time.sleep(0.1)

            # 批处理逻辑
    def process_batch_parallel_with_chunk(self):
        """后台线程，用于批量处理请求"""
        global request_queue, request_text, results, request_past_key_values
        while True:
            start_time = time.time()
            task = None

            with queue_lock:  # 使用锁来保护 request_queue
                while not task and (time.time() - start_time) < MAX_WAIT_TIME:
                    if request_queue:
                        request = request_queue.pop(0)
                        req_id = request["request_id"]
                        job_type = request["request_type"]
                        job_text = request["text"]
                        gamma = request["gamma"]
                        task = (req_id, job_type, job_text, gamma)
                    else:
                        time.sleep(0.1)

            if task:
                # print(f"task: {task}")
                text = task[2]
                past_key_values = request_past_key_values.get(req_id, None)
                past_sequence_length = past_key_values[0][0].shape[2] if past_key_values else 0
                
                total_input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                input_ids = total_input_ids if past_key_values is None else total_input_ids[:, past_sequence_length:]
                
                # print(f"input_ids: {input_ids}, input_length: {input_ids.shape}, past_sequence_length: {past_sequence_length}")
                # input_ids = inputs.input_ids.to(self.device)
                # attention_mask = inputs.attention_mask.to(self.device)

                with torch.no_grad():
                    Mp = self.model(
                            input_ids=input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                        ) 
                target_logits = Mp.logits
                past_key_values = Mp.past_key_values
                verified_text = text 
                
                job_type = task[1]
                if job_type == "init":
                    print(f"init request id: {task[0]}, input_text_shape: {input_ids.shape}")
                    next_token_logits = target_logits[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    output = torch.cat((total_input_ids[0], next_token_id), dim=0)
                    request_past_key_values[req_id] = past_key_values

                    generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    verified_text = generated_text
                    #print(f"initialized text: {verified_text}")
                elif job_type == "verify":
                    generate_step = int(task[3])
                    if generate_step == 0:
                        print(f"verify change to init request id: {task[0]}, input_text_shape: {input_ids.shape}")
                        next_token_logits = target_logits[0, -1, :]
                        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                        output = torch.cat((total_input_ids[0], next_token_id), dim=0)
                        request_past_key_values[req_id] = past_key_values

                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        verified_text = generated_text
                    else:
                        print(f"generate_step: {generate_step}, draft_length: {input_ids.shape[1]}")
                        #assert(generate_step == input_ids.shape[1])
                        #print(f"verify request id: {task[0]}, input_text: {text}, gamma: {generate_step}")
                        predicted_token_ids = target_logits[0][-generate_step-1:, :].argmax(dim=-1)
                        n = generate_step
                        for j in range(generate_step):
                            if torch.equal(predicted_token_ids[j], input_ids[0][j+1]):
                                continue
                            else:
                                n = j
                                break
                        if n == generate_step:
                            request_past_key_values[req_id] = past_key_values
                            verified_text = self.tokenizer.decode(total_input_ids[0], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)
                        else:
                            truncated_past_key_values = []
                            for layer in past_key_values:
                                key, value = layer
                                key = key[:, :, :-generate_step+n, :]
                                value = value[:, :, :-generate_step+n, :]
                                truncated_past_key_values.append((key, value))
    
                            # Update past_key_values for the request
                            request_past_key_values[req_id] = truncated_past_key_values
                            verified_text = self.tokenizer.decode(total_input_ids[0][:-generate_step+n], skip_special_tokens=True) + self.tokenizer.decode(predicted_token_ids[n], skip_special_tokens=True)
                        #print(f"Verified text: {verified_text}")
                        print(f"Matched length: {n}, draft length: {generate_step}, passed rate: {n/generate_step}")

                with condition:
                    with queue_lock:  # 加锁来保护 results
                        results[req_id] = verified_text
                        request_text[req_id] = verified_text
                        condition.notify_all()
            else:
                time.sleep(0.1)
# 服务器设置
def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    batch_pb2_grpc.add_BatchServiceServicer_to_server(VerficationServer(args), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051.")

    server.wait_for_termination()

if __name__ == '__main__':
    args = parse_arguments()
    serve(args)