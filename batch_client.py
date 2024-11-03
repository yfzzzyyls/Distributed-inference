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
        device_id = f"cuda:{self.accelerator.process_index}"
    
        # 使用 to() 将模型加载到指定的 GPU
        self.model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, torch_dtype=torch.float16).to(device_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model)
        self.device = torch.device(device_id)  # 设置设备为指定 GPU

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
    

    def add_async_request(self, request_id, request_type, text):
        """发送请求到服务器，不阻塞等待返回已验证文本"""

        # 创建请求对象
        request = batch_pb2.Request(request_id=request_id, request_type=request_type, text=text)

        # 使用异步 future 方法发送请求
        future = self.stub.AddRequest.future(request)

        # 定义一个回调函数来处理返回结果
        def handle_response(future):
            try:
                # 这里可以捕获请求的返回值
                response = future.result()  # 获取异步调用结果
                verified_text = response.verified_text
                print(f"请求 {request_id} 的验证文本已返回: {verified_text}")
                # 在此处执行进一步的处理逻辑
            except Exception as e:
                # 如果请求失败，可以捕获异常
                print(f"请求 {request_id} 出现异常: {e}")
            # 设置回调函数，当 future 完成时调用 handle_response

        future.add_done_callback(handle_response)

        # 返回 future 对象，如果需要进一步处理或等待可以使用
        return future

    def close(self):
        """关闭与服务器的连接"""
        self.channel.close()

    def speculative_decoding(self, prompt, max_length=20, generate_step=4):
        """使用 speculative decoding 生成文本"""
        generate_step = self.args.gamma
        max_length = self.args.max_tokens

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
            #print(f"Passed length: {passed_length}")
            length += passed_length
            if self.use_cache:
                if 0 < passed_length < generate_step+1:
                    #print(f"Last Past key values: {past_key_values[0][0].shape}")
                    past_key_values = tuple(
                                        tuple(
                                            tensor[:, :, :-generate_step-1 + passed_length, :] 
                                            for tensor in inner_tuple
                                        )
                                        for inner_tuple in past_key_values
                                    )
                    #print(f"Cutted Past key values: {past_key_values[0][0].shape}")
                if passed_length > 0:
                    input_ids = input_ids[:, -1:]
                    #print(input_ids)
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
                
            verified_text = self.add_request(request_id, "verify", str(generate_step))
            print(f"Verified Successfully.")
            timestamps.append(time.time())
            #print(f"Verified text: {verified_text}")
        print(f"final text: {verified_text}")
        self.add_request(request_id, "delete", " "+str(length))
        return verified_text, timestamps
        
    def speculative_decoding_parallel(self, prompt, max_length=20, generate_step=4):
        """使用 speculative decoding 生成文本"""
        generate_step = self.args.gamma
        max_length = self.args.max_tokens

        timestamps = []
        timestamps.append(time.time())
        request_id = str(uuid.uuid4().hex)
        # 模拟发送请求
        length = 0
        verified_text_future = self.add_async_request(request_id, "init", prompt)
        input_text = prompt
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        old_input_ids = input_ids
        old_length = len(old_input_ids[0])
        past_key_values = None
        cur_mode = True

        while length < max_length:
            draft_ids = input_ids.new_empty((0,))
            draft_outputs = ""

            while not verified_text_future.done(): 
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
                
                draft_ids = torch.cat((draft_ids, xi), dim=1).to(self.device)
                draft_outputs += self.tokenizer.decode(xi[0], skip_special_tokens=True)
                timestamps.append(time.time()) 

            verified_text = verified_text_future.result().verified_text
            verified_ids = self.tokenizer.encode(verified_text, return_tensors='pt').to(self.device)
            if cur_mode:
                #print(f"verified text: {verified_text}")
                #print(f"Draft output: {draft_outputs}")
                if verified_ids[0][-1] == draft_ids[0][0]:
                    print(f"cur_mode")
                    cur_mode = False
                    input_ids = input_ids.to(self.device)
                    draft_outputs = self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)
                    print(f"Draft output: {draft_outputs}")
                    self.add_request(request_id, "update", draft_outputs) 
                    verified_text_future = self.add_async_request(request_id, "verify", str(len(draft_ids[0])-1))
                else:
                    print(f"cur_mode conitnue")
                    input_text = verified_text
                    input_ids = verified_ids
                    verified_text_future = self.add_async_request(request_id, "init", verified_text)
            else:
                #print(f"verified text: {verified_text}")
                #print(f"Draft output: {draft_outputs}")
                #print(f"input_text: {input_text}")
                print(f"verified_text: {verified_ids[0][-5:]}, old_: {old_input_ids[0][-5:]}")
                if torch.equal(verified_ids[0][:-1], old_input_ids[0]) and verified_ids[0][-1]  == draft_ids[0][0]:
                    print(f"not cur_mode")
                    input_ids = torch.cat((verified_ids, draft_ids[:,1:]), dim=1).to(self.device)
                    self.add_request(request_id, "update", self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)) 
                    verified_text_future = self.add_async_request(request_id, "verify", str(len(draft_ids[0])-1))
                else:
                    print(f"not cur_mode failue")
                    input_ids = verified_ids
                    cur_mode = True
                    verified_text_future = self.add_async_request(request_id, "init", verified_text)
            
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"Next forward input_ids: {self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            # input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            timestamps.append(time.time())

            new_length = len(input_ids[0])
            passed_length = new_length - old_length
            old_input_ids = input_ids 
            old_length = new_length
            #print(f"Passed length: {passed_length}")
            length += passed_length
              
        while not verified_text_future.done():
            pass
        # 等待处理完成
        print(f"final text: {input_text}")
        self.add_request(request_id, "delete", " "+str(length))
        return verified_text, timestamps

    def speculative_decoding_parallel_with_chunked(self, prompt, max_length=20, generate_step=4):
        """使用 speculative decoding 生成文本"""
        generate_step = self.args.gamma
        max_length = self.args.max_tokens

        timestamps = []
        timestamps.append(time.time())
        request_id = str(uuid.uuid4().hex)
        # 模拟发送请求
        length = 0
        verified_text_future = self.add_async_request(request_id, "init", prompt)
        input_text = prompt
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        old_input_ids = input_ids
        old_length = len(old_input_ids[0])
        past_key_values = None
        cur_mode = True

        while length < max_length:
            draft_ids = input_ids.new_empty((0,))
            draft_outputs = ""

            while not verified_text_future.done(): 
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
                
                draft_ids = torch.cat((draft_ids, xi), dim=1).to(self.device)
                draft_outputs += self.tokenizer.decode(xi[0], skip_special_tokens=True)
                timestamps.append(time.time()) 

            verified_text = verified_text_future.result().verified_text
            verified_ids = self.tokenizer.encode(verified_text, return_tensors='pt').to(self.device)
            if cur_mode:
                #print(f"verified text: {verified_text}")
                #print(f"Draft output: {draft_outputs}")
                if verified_ids[0][-1] == draft_ids[0][0]:
                    print(f"cur_mode")
                    cur_mode = False
                    input_ids = input_ids.to(self.device)
                    draft_outputs = self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)
                    print(f"Draft output: {draft_outputs}")
                    if len(draft_ids[0]) > 1:
                        self.add_request(request_id, "update", self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)) 
                        verified_text_future = self.add_async_request(request_id, "verify", str(len(draft_ids[0])-1))
                    else:
                        verified_text_future = self.add_async_request(request_id, "init", verified_text)
                else:
                    print(f"cur_mode conitnue")
                    input_text = verified_text
                    input_ids = verified_ids
                    verified_text_future = self.add_async_request(request_id, "init", verified_text)
            else:
                #print(f"verified text: {verified_text}")
                #print(f"Draft output: {draft_outputs}")
                #print(f"input_text: {input_text}")
                print(f"verified_text: {verified_ids[0][-5:]}, old_: {old_input_ids[0][-5:]}")
                if verified_ids[0][-1]  == draft_ids[0][0]:
                    print(f"not cur_mode")
                    input_ids = torch.cat((verified_ids, draft_ids[:,1:]), dim=1).to(self.device)
                    if len(draft_ids[0]) > 1:
                        self.add_request(request_id, "update", self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)) 
                        verified_text_future = self.add_async_request(request_id, "verify", str(len(draft_ids[0])-1))
                    else:
                        verified_text_future = self.add_async_request(request_id, "init", verified_text)
                else:
                    print(f"not cur_mode failue")
                    input_ids = verified_ids
                    cur_mode = True
                    verified_text_future = self.add_async_request(request_id, "init", verified_text)
            
            input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"Next forward input_ids: {self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            # input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            timestamps.append(time.time())

            new_length = len(input_ids[0])
            passed_length = new_length - old_length
            old_input_ids = input_ids 
            old_length = new_length
            #print(f"Passed length: {passed_length}")
            length += passed_length
              
        while not verified_text_future.done():
            pass
        # 等待处理完成
        print(f"final text: {input_text}")
        self.add_request(request_id, "delete", " "+str(length))
        return verified_text, timestamps

# 示例使用
if __name__ == "__main__":
    # 初始化客户端
    args = parse_arguments()
    client = BatchClient(args)
    ans, timestamps = client.speculative_decoding_parallel_with_chunked("The quick brown fox jumps over the lazy dog.")
    print(f"timestamps: {timestamps}")
