import grpc
import protos.batch_pb2 as batch_pb2
import protos.batch_pb2_grpc as batch_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from accelerate import Accelerator
import torch
import time
import uuid
from utils import parse_arguments


class BatchClient:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()

        # 连接到 gRPC 服务器
        print(f"process number: {self.accelerator.num_processes}")

        self.channel = grpc.insecure_channel(f'{self.args.host}:{self.args.port}')
        self.stub = batch_pb2_grpc.BatchServiceStub(self.channel)
        self.use_cache = self.args.use_cache
        self.load_model()

    def load_model(self):
        quantization_config = QuantoConfig(weights="int4")
        self.model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, torch_dtype=torch.float16)  # 替换成实际的模型路径
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model)
        self.device = torch.device(f"cuda:{self.accelerator.process_index}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = "left"

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
        old_length = len(self.tokenizer.encode(verified_text, return_tensors='pt').to(self.device)[0])
        past_key_values = None

        while length < max_length:
            input_ids = self.tokenizer.encode(verified_text, return_tensors='pt').to(self.device)
            new_length = len(input_ids[0])
            passed_length = new_length - old_length
            print(f"Passed length: {passed_length}")
            length += passed_length
            if self.use_cache:
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
                    Mq = self.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=self.use_cache,
                    )
                past_key_values = Mq.past_key_values
                draft_logits = Mq.logits[..., -1, :]
                xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
                if self.use_cache:
                    input_ids = xi.to(self.device)
                else:
                    input_ids = torch.cat((input_ids, xi), dim=1).to(self.device)

                draft_output = self.tokenizer.decode(xi[0], skip_special_tokens=True)
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
    args = parse_arguments()
    client = BatchClient(args)
    ans, timestamps = client.speculative_decoding("The quick brown fox jumps over the lazy dog.")
    print(f"timestamps: {timestamps}")
