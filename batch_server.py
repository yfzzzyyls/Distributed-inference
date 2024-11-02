import time
import threading
import grpc
from concurrent import futures
import protos.batch_pb2 as batch_pb2
import protos.batch_pb2_grpc as batch_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch
from utils import parse_arguments


# 全局变量
request_queue = []
request_text = {}
results = {}
condition = threading.Condition()
BATCH_SIZE = 2
MAX_WAIT_TIME = 2
generate_step = 4


# 实现服务
class VerficationServer(batch_pb2_grpc.BatchServiceServicer):
    def __init__(self, args):
        self.args = args
        self.load_model()
        # 启动批处理线程
        self.batch_thread = threading.Thread(target=self.process_batch, daemon=True)
        self.batch_thread.start()

    def load_model(self):
        quantization_config = QuantoConfig(weights="int4")
        self.model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.float16)  # 替换成实际的模型路径
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
        with condition:
            # 处理请求的不同类型
            if request.request_type == "init":
                if request.request_id not in request_text:
                    request_queue.append({"request_id": request.request_id, "text": request.text, "request_type": "init"})
                    request_text[request.request_id] = request.text
                    print(f"Initialized request {request.request_id}: {request.text}")
                else:
                    print(f"Request {request.request_id} already initialized.")
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
                    request_queue.append({"request_id": request.request_id, "text": request_text[request.request_id], "request_type": "verify"})
                print(f"Verification request {request.request_id} added to batch queue.")
            elif request.request_type == "delete":
                if request.request_id in request_text:
                    del request_text[request.request_id]
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
            condition.wait_for(lambda: request.request_id in results)
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
                    batch.append((req_id, job_type, job_text))
                else:
                    time.sleep(0.1)  # 无请求时稍作等待

            if batch:
                # 批量处理请求
                batch_size = len(batch)
                texts = [req for _, _, req in batch]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                print(f"Processing batch: {texts}, {input_ids}")

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
                        predicted_token_ids = target_logits[i][-generate_step-1:, :].argmax(dim=-1)

                        # 解码预测的 token 和对应输入的 token
                        predicted_text = self.tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
                        draft_text = self.tokenizer.decode(input_ids[i][-generate_step:], skip_special_tokens=True)
                        n = generate_step
                        print(f"predicted_text: {predicted_text}, draft_text: {draft_text}")
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
                        print(f"Sample {i+1}")
                        print("Match:", verified_text[i])

                # 存储批处理结果并通知等待的请求
                with condition:
                    for idx, (req_id,_, _) in enumerate(batch):
                        results[req_id] = verified_text[idx]  # 将每个请求的结果存入字典
                        request_text[req_id] = verified_text[idx]  # 更新请求文本
                        print(f"Processed result for {req_id}: {results[req_id]}")
                    condition.notify_all()  # 通知所有等待的请求
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