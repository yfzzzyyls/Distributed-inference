import grpc
import protos.batch_pb2 as batch_pb2
import protos.batch_pb2_grpc as batch_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
import torch
import time
import uuid

# 加载模型和 tokenizer
quantization_config = QuantoConfig(weights="int4")
model_path = "/home/apc/llama/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path)  # 替换成实际的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = "left"

use_cache = False

class BatchClient:
    def __init__(self, host='localhost', port=50051):
        # 连接到 gRPC 服务器
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = batch_pb2_grpc.BatchServiceStub(self.channel)

    def add_request(self, request_id, request_type, text):
        """发送请求到服务器并等待返回已验证文本"""
        request = batch_pb2.Request(request_id=request_id, request_type=request_type, text=text)
        
        # 直接调用 AddRequest 并等待返回已验证文本
        response = self.stub.AddRequest(request)
        
        # 输出或返回 verified_text 结果
        # print(f"Received verified text for {request_id}: {response.verified_text}")
        return response.verified_text

    def close(self):
        """关闭与服务器的连接"""
        self.channel.close()

    def speculative_decoding(self, prompt, max_length=20, generate_step=4):
        """使用 speculative decoding 生成文本"""
        timestamps = []
        timestamps.append(time.time())
        request_id = str(uuid.uuid4().hex)
        # 模拟发送请求
        length = 0
        verified_text = self.add_request(request_id, "init", prompt)
        timestamps.append(time.time())
        print(f"Initialized text: {verified_text}")
        old_length = len(tokenizer.encode(verified_text, return_tensors='pt').to(device)[0])
        past_key_values = None

        while length < max_length:
            input_ids = tokenizer.encode(verified_text, return_tensors='pt').to(device)
            new_length = len(input_ids[0])
            passed_length = new_length - old_length
            print(f"Passed length: {passed_length}")
            length += passed_length
            if use_cache:
                if 0 < passed_length < generate_step+1:
                    print(f"Last Past key values: {past_key_values[0][0].shape}")
                    past_key_values = tuple(
                                        tuple(
                                            tensor[:, :, :-generate_step-1 + passed_length, :] 
                                            for tensor in inner_tuple
                                        )
                                        for inner_tuple in past_key_values
                                    )
                    print(f"Cutted Past key values: {past_key_values[0][0].shape}")
                if passed_length > 0:
                    input_ids = input_ids[:, -1:]
                    print(input_ids)
            old_length = new_length
            for _ in range(generate_step):
                with torch.no_grad():
                    Mq = model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                    )
                past_key_values = Mq.past_key_values
                draft_logits = Mq.logits[..., -1, :]
                xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
                if use_cache:
                    input_ids = xi.to(device)
                else:
                    input_ids = torch.cat((input_ids, xi), dim=1).to(device)

                draft_output = tokenizer.decode(xi[0], skip_special_tokens=True)
                print(f"Draft output: {draft_output}")
                self.add_request(request_id, "update", draft_output)   
                timestamps.append(time.time())      
                
            verified_text = self.add_request(request_id, "verify", "")
            timestamps.append(time.time())
            print(f"Verified text: {verified_text}")
            

        # 等待处理完成
        print(f"final text: {verified_text}")

        self.add_request(request_id, "delete", " "+str(length))

        return verified_text, timestamps



# 示例使用
if __name__ == "__main__":
    # 初始化客户端
    client = BatchClient(host="localhost", port=50051)
    
    ans, timestamps = client.speculative_decoding("The quick brown fox jumps over the lazy dog.")

    print(f"timestamps: {timestamps}")
