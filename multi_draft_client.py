import grpc
import protos.service_pb2 as batch_pb2
import protos.service_pb2_grpc as batch_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
from accelerate import Accelerator
import torch
import time
import uuid
from utils import parse_arguments, rollback_past_key_values


class DraftClient:
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
    
    def add_init_request(self, request_id, text):
        """发送请求到服务器并等待返回已验证文本"""
        request = batch_pb2.InitRequest(request_id=request_id, request_type="init", text=text)
        
        # 直接调用 AddRequest 并等待返回已验证文本
        response = self.stub.InitRequest(request)
        
        # 输出或返回 verified_text 结果
        # print(f"Received verified text for {request_id}: {response.verified_text}")
        return response.next_text, response.generate_finished
    
    def add_verify_request(self, request_id, text):
        """发送请求到服务器并等待返回已验证文本"""
        request = batch_pb2.VerifyRequest(request_id=request_id, request_type="verify", text=text)
        
        # 直接调用 AddRequest 并等待返回已验证文本
        response = self.stub.VerifyRequest(request)
        
        # 输出或返回 verified_text 结果
        # print(f"Received verified text for {request_id}: {response.verified_text}")
        return response.verified_text, response.passed_tokens, response.generate_finished
    
    def add_delete_request(self, request_id):
        """发送请求到服务器并等待返回已验证文本"""
        request = batch_pb2.DeleteRequest(request_id=request_id, request_type="delete")
        
        # 直接调用 AddRequest 并等待返回已验证文本
        self.stub.DeleteRequest(request)
        
        # 输出或返回 verified_text 结果
        # print(f"Received verified text for {request_id}: {response.verified_text}")
        return
    
    def add_async_request(self, request_id, request_type, text):
        """Send a request to the server asynchronously, without blocking for the verified text."""
        if request_type == "init":
            request = batch_pb2.InitRequest(request_id=request_id, request_type=request_type, text=text)
            future = self.stub.InitRequest.future(request)

        elif request_type == "verify":
            request = batch_pb2.VerifyRequest(request_id=request_id, request_type=request_type, text=text)
            future = self.stub.VerifyRequest.future(request)

        # Define a callback function to handle the response
        def handle_response(future):
            try:
                # Retrieve the asynchronous call result
                response = future.result()

                # Process response based on request type
                if request_type == "init":
                    next_text, generate_finished = response.next_text, response.generate_finished
                    # Additional processing if needed
                    # print(f"Received init response for {request_id}: {next_text}, {generate_finished}")

                elif request_type == "verify":
                    verified_text = response.verified_text
                    passed_tokens = response.passed_tokens
                    generate_finished = response.generate_finished
                    # Additional processing if needed
                    # print(f"Received verify response for {request_id}: {verified_text}")

            except Exception as e:
                # Catch any errors that occur during the request
                print(f"Request {request_id} encountered an exception: {e}")

        # Attach the callback to the future result
        future.add_done_callback(handle_response)

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
                if self.tokenizer.eos_token_id == xi[0][0]:
                    break

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
        max_length = 200
        timestamps = []
        timestamps.append(time.time())
        request_id = str(uuid.uuid4().hex)
        generate_finished = False

        result_ids = torch.full((1, max_length), 0, dtype=torch.long).to(self.device)
        current_length = 0
        
        next_text_future = self.add_async_request(request_id, "init", prompt)
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        original_input_ids = input_ids
        
        old_length = input_ids.shape[1]
        draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
        
        past_key_values = None
        cur_mode = True

        
        draft_num = 0

        while current_length < max_length and not generate_finished:
            print(f"Current length: {current_length}")
            # draft_ids = input_ids.new_empty((0,))
            # draft_outputs = ""

            while not next_text_future.done() and draft_num < generate_step: 
                print(f"draft generate loop: input ids shape: {input_ids.shape}")
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
                # draft_outputs += self.tokenizer.decode(xi[0], skip_special_tokens=True)
                timestamps.append(time.time()) 
                draft_num += 1
                if self.tokenizer.eos_token_id == xi[0][0]:
                    break
            print(f"draft num: {draft_num}")
            verification_service_response = next_text_future.result()
            if verification_service_response.generate_finished:
                generate_finished = True
                print(f"generate finished")
                break
            next_text = verification_service_response.next_text
            next_ids = self.tokenizer.encode(next_text, return_tensors='pt').to(self.device)
            print(f"Next text: {next_text}")
            if cur_mode:
                result_ids[0][current_length] = next_ids[0][0]
                current_length += 1
                print(f"init mode draft_ids shape: {draft_ids.shape}")

                if draft_num == 0:
                    input_ids = next_ids
                    print(f"no enough draft, continue to init")
                    next_text_future = self.add_async_request(request_id, "init", None)

                elif next_ids[0][0] == draft_ids[0][0]:
                    print(f"init to verify")
                    #print(f"cur_mode")
                    # input_ids = input_ids.to(self.device)
                    # draft_outputs = self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True)
                    #print(f"Draft output: {draft_outputs}")
                    if draft_num > 1:
                        cur_mode = False
                        input_ids = draft_ids[:,-1:]
                        draft_ids = draft_ids[:,1:]
                        draft_num -= 1
                        # self.add_request(request_id, "update", self.tokenizer.decode(draft_ids[0][1:], skip_special_tokens=True))
                        draft_text = self.tokenizer.decode(draft_ids[0], skip_special_tokens=True) 
                        print(f"verify draft: {draft_text}")
                        next_text_future = self.add_async_request(request_id, "verify", draft_text)
                    else:
                        input_ids = next_ids
                        draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                        draft_num = 0
                        print(f"no enough draft, continue to init")
                        next_text_future = self.add_async_request(request_id, "init", None)
                else:
                    print(f"stay init")
                    #print(f"cur_mode conitnue")
                    input_text = next_text
                    input_ids = next_ids
                    draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                    past_key_values = rollback_past_key_values(past_key_values, draft_num)
                    draft_num = 0
                    next_text_future = self.add_async_request(request_id, "init", None)
            else:
                passed_token_num = next_text_future.result().passed_tokens
                result_ids[0][current_length:current_length+passed_token_num] = draft_ids[0][:passed_token_num]
                if current_length+passed_token_num < max_length:
                    result_ids[0][current_length+passed_token_num] = next_ids[0][0]
                current_length += passed_token_num + 1
                print(f"Verification retured passed token number: {passed_token_num}, draft_ids shape: {draft_ids.shape}")
                #print(f"verified text: {verified_text}")
                #print(f"Draft output: {draft_outputs}")
                #print(f"input_text: {input_text}")
                #print(f"verified_text: {verified_ids[0][-5:]}, old_: {old_input_ids[0][-5:]}")
                if passed_token_num == draft_num:
                    print(f"no enough draft, continue to init")
                    cur_mode = True
                    input_ids = next_ids
                    draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                    draft_num = 0
                    next_text_future = self.add_async_request(request_id, "init", None)
                else:
                    if next_ids[0][0]  == draft_ids[0][passed_token_num]:
                        #print(f"not cur_mode")
                        if draft_num > passed_token_num + 1:
                            print(f"stay verify")
                            input_ids = draft_ids[:, -1:]
                            draft_ids = draft_ids[:,passed_token_num+1:]
                            draft_num -= (passed_token_num + 1)
                            draft_text = self.tokenizer.decode(draft_ids[0], skip_special_tokens=True) 
                            print(f"verify draft: {draft_text}")
                            next_text_future = self.add_async_request(request_id, "verify", draft_text)
                        else:
                            print(f"no enough draft, continue to init")
                            cur_mode = True
                            input_ids = next_ids
                            draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                            draft_num = 0
                            next_text_future = self.add_async_request(request_id, "init", None)
                    else:
                        print(f"verify to init")
                        #print(f"not cur_mode failue")
                        input_ids = next_ids
                        cur_mode = True
                        draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                        past_key_values = rollback_past_key_values(past_key_values, draft_num)
                        draft_num = 0 
                        next_text_future = self.add_async_request(request_id, "init", None)


                # if next_ids[0][0]  == draft_ids[0][passed_token_num+1]:
                #     print(f"stay verify")
                #     #print(f"not cur_mode")
                #     if len(draft_ids[0]) > 1:
                #         input_ids = draft_ids[0][-1:]
                #         draft_ids = draft_ids[:,passed_token_num+2:]
                #         draft_text = self.tokenizer.decode(draft_ids[0], skip_special_tokens=True) 
                #         print(f"verify draft: {draft_text}")
                #         next_text_future = self.add_async_request(request_id, "verify", draft_text)
                #     else:
                #         print(f"no enough draft, continue to init")
                #         cur_mode = True
                #         input_ids = next_ids
                #         draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                #         next_text_future = self.add_async_request(request_id, "init", None)
                # else:
                #     print(f"verify to init")
                #     #print(f"not cur_mode failue")
                #     input_ids = next_ids
                #     cur_mode = True
                #     draft_ids = torch.empty((1, 0), dtype=torch.long).to(self.device)
                #     past_key_values = rollback_past_key_values(past_key_values, past_key_values[0][0].shape[2] - old_length - passed_token_num) 
                #     next_text_future = self.add_async_request(request_id, "init", None)

            # input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # print(f"Next forward input_ids: {self.tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
            # input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            timestamps.append(time.time())
            # old_input_ids = input_ids 
            #print(f"Passed length: {passed_length}")
              
        # 等待处理完成
        print(f"final text shape: {result_ids.shape}")
        result_ids = torch.cat((original_input_ids, result_ids), dim=1)
        final_text = self.tokenizer.decode(result_ids[0], skip_special_tokens=True)
        print(f"final text: {self.tokenizer.decode(result_ids[0], skip_special_tokens=True)}")
        self.add_delete_request(request_id)
        return final_text, timestamps

# 示例使用
if __name__ == "__main__":
    # 初始化客户端
    args = parse_arguments()
    client = DraftClient(args)
    ans, timestamps = client.speculative_decoding_parallel_with_chunked("The quick brown fox jumps over the lazy dog.")
    print(f"timestamps: {timestamps}")
