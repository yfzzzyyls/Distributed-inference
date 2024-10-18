import grpc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import protos.model_service_pb2
import protos.model_service_pb2_grpc
import time
import uuid
import gevent
import concurrent.futures

class ModelServiceClient:
    def __init__(self, model_name=None, model=None, tokenizer=None, server_address='localhost:2024', quantize=False, max_length=50, generate_step=6, debug_mode=False):
        """
        初始化 gRPC 客户端，支持通过 model_name 加载模型或直接传入预加载的模型和 tokenizer。
        """
        self.server_address = server_address
        self.max_length = max_length
        self.generate_step = generate_step
        self.generated_uuid = str(uuid.uuid4())
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=generate_step)
        self.debug_mode = debug_mode

        # 如果没有传入模型和 tokenizer，则根据 model_name 来加载
        if model is None or tokenizer is None:
            self.quantize = QuantoConfig(weights="int4") if quantize else None
            self.tokenizer, self.model, self.vocab_size = self.load_model_and_tokenizer(model_name)
        else:
            # 直接使用传入的模型和 tokenizer
            self.model = model
            self.tokenizer = tokenizer
            self.vocab_size = model.config.vocab_size

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 初始化 gRPC 通道和存根
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = protos.model_service_pb2_grpc.ModelServiceStub(self.channel)

    def load_model_and_tokenizer(self, model_name):
        """
        加载模型和 tokenizer，并返回模型、tokenizer 和词汇表大小。
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=self.quantize)
        model.eval()  # 设置为评估模式
        vocabulary_size = model.config.vocab_size
        return tokenizer, model, vocabulary_size

    def update_token(self, k, draft_output):
        """
        更新生成的 token
        """
        token_update_request = protos.model_service_pb2.UpdateTokenRequest(
            user_uuid=self.generated_uuid,
            index=k,
            input_text=draft_output,
        )
        return self.stub.UpdateToken(token_update_request)

    #async def process_step(self, queue, step_index, draft_output):
    #    """
    #    异步处理生成的 token 并发送更新请求
    #    """
    #    result = await asyncio.to_thread(self.update_token, step_index, draft_output)
    #    await queue.put((step_index, result))

    def speculative_decode_gevent(self, prompt, generate_step=None):
        """
        与 gRPC 服务进行交互，执行 token 验证和生成任务（speculative 模式）。
        """
        if generate_step is None:
            generate_step = self.generate_step

        drafter_cache = None
        total_generated = 0  # 已生成的 tokens 数
        first_target = True
        output_text = ""
        start_time = time.time()  # 记录开始时间

        if first_target:
            prepare_request = protos.model_service_pb2.PrepareSpeculativeRequest(
                user_uuid=self.generated_uuid,
                prompt=prompt,
                max_length=self.max_length,
                generate_step=generate_step,
                exact_mode=True,
                debug_mode=True
            )
            prepare_response = self.stub.PrepareSpeculative(prepare_request)
            first_tokens = prepare_response.first_tokens  # 这是一个 token ID 列表
            output_text = first_tokens
            total_generated += 1

        input_ids = self.tokenizer.encode(output_text, return_tensors='pt').to(self.device)

        while total_generated < self.max_length:
            q = torch.zeros((1, generate_step, self.vocab_size), device=self.device)
            total_generated = len(input_ids[0])
            if total_generated >= self.max_length:
                break
            
            tasks = []
            draft_step = 0
            draft_output = [None for _ in range(generate_step)]
            for k in range(generate_step):
                with torch.no_grad():
                    Mq = self.model(
                        input_ids=input_ids,
                        past_key_values=drafter_cache,
                        use_cache=False,
                    )
                drafter_cache = Mq.past_key_values
                draft_logits = Mq.logits[..., -1, :]
                q[0, k] = draft_logits.to(self.device)
                xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat((input_ids, xi), dim=1).to(self.device)

                draft_output[k] = self.tokenizer.decode(xi[0], skip_special_tokens=True)
                tasks.append(gevent.spawn(self.update_token, k, draft_output[k]))                
                draft_step += 1

            gevent.joinall(tasks)
            token_request = protos.model_service_pb2.VerifyTokensRequest(
                user_uuid=self.generated_uuid
            )
            token_response = self.stub.VerifyTokens(token_request)

            if token_response.finished:
                break

            passed_tokens = token_response.passed_tokens
            verified_tokens = self.tokenizer.encode(token_response.verified_tokens, return_tensors='pt').to(self.device)
            if passed_tokens < generate_step:
                input_ids = torch.cat((input_ids[0][:-generate_step + passed_tokens].unsqueeze(0), verified_tokens), dim=1).to(self.device)
            else:
                input_ids = torch.cat((input_ids, verified_tokens), dim=1).to(self.device)

        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Speculative with gevent 生成过程执行时间：{end_time - start_time} 秒")
        return output_text
    
    def speculative_decode_thread(self, prompt, max_length=None):
        """
        与 gRPC 服务进行交互，执行 token 验证和生成任务（speculative 模式）。
        """
        if max_length is None:
            max_length = self.max_length

        drafter_cache = None
        total_generated = 0  # 已生成的 tokens 数
        first_target = True
        output_text = ""
        start_time = time.time()  # 记录开始时间

        if first_target:
            prepare_request = protos.model_service_pb2.PrepareSpeculativeRequest(
                user_uuid=self.generated_uuid,
                prompt=prompt,
                max_length=self.max_length,
                generate_step=self.generate_step,
                exact_mode=True,
                debug_mode=self.debug_mode
            )
            prepare_response = self.stub.PrepareSpeculative(prepare_request)
            first_tokens = prepare_response.first_tokens  # 这是一个 token ID 列表
            output_text = first_tokens
            total_generated += 1

        input_ids = self.tokenizer.encode(output_text, return_tensors='pt').to(self.device)

        while total_generated < self.max_length:
            #q = torch.zeros((1, self.generate_step, self.vocab_size), device=self.device)
            total_generated = len(input_ids[0])
            if total_generated >= self.max_length:
                break
            
            tasks = []
            draft_step = 0
            draft_output = [None for _ in range(self.generate_step)]
            for k in range(self.generate_step):
                with torch.no_grad():
                    Mq = self.model(
                        input_ids=input_ids,
                        past_key_values=drafter_cache,
                        use_cache=False,
                    )
                drafter_cache = Mq.past_key_values
                draft_logits = Mq.logits[..., -1, :]
                #q[0, k] = draft_logits.to(self.device)
                xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat((input_ids, xi), dim=1).to(self.device)
                # input_ids = xi
                
                draft_output[k] = self.tokenizer.decode(xi[0], skip_special_tokens=True)
                tasks.append(self.executor.submit(self.update_token, k, draft_output[k]))                
                draft_step += 1
            
            concurrent.futures.wait(tasks)
            token_request = protos.model_service_pb2.VerifyTokensRequest(
                user_uuid=self.generated_uuid
            )
            token_response = self.stub.VerifyTokens(token_request)

            if token_response.finished:
                break

            passed_tokens = token_response.passed_tokens
            #print(f"passed tokens: {token_response.verified_tokens}")
            verified_tokens = self.tokenizer.encode(token_response.verified_tokens, return_tensors='pt').to(self.device)
            if passed_tokens < self.generate_step:
                #drafter_cache = tuple(
    #(key[:, :, :-self.generate_step + passed_tokens, :], value[:, :, :-self.generate_step + passed_tokens, :]) for key, value in drafter_cache
#)
                #input_ids = verified_tokens
                input_ids = torch.cat((input_ids[0][:-self.generate_step + passed_tokens].unsqueeze(0), verified_tokens), dim=1).to(self.device)
            else:
                #input_ids = verified_tokens
                input_ids = torch.cat((input_ids, verified_tokens), dim=1).to(self.device)
            

        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Speculative with thread 生成过程执行时间：{end_time - start_time} 秒")
        return output_text
    
    def speculative_decode(self, prompt, max_length=None):
        """
        与 gRPC 服务进行交互，执行 token 验证和生成任务（speculative 模式）。
        """
        if max_length is None:
            max_length = self.max_length

        drafter_cache = None
        total_generated = 0  # 已生成的 tokens 数
        first_target = True
        output_text = ""
        start_time = time.time()  # 记录开始时间

        if first_target:
            prepare_request = protos.model_service_pb2.PrepareSpeculativeRequest(
                user_uuid=self.generated_uuid,
                prompt=prompt,
                max_length=self.max_length,
                generate_step=self.generate_step,
                exact_mode=True,
                debug_mode=True
            )
            prepare_response = self.stub.PrepareSpeculative(prepare_request)
            first_tokens = prepare_response.first_tokens  # 这是一个 token ID 列表
            output_text = first_tokens
            total_generated += 1

        input_ids = self.tokenizer.encode(output_text, return_tensors='pt').to(self.device)

        while total_generated < max_length:
            q = torch.zeros((1, self.generate_step, self.vocab_size), device=self.device)
            total_generated = len(input_ids[0])
            if total_generated >= max_length:
                break
            
            tasks = []
            draft_step = 0
            draft_output = [None for _ in range(self.generate_step)]
            for k in range(self.generate_step):
                with torch.no_grad():
                    Mq = self.model(
                        input_ids=input_ids,
                        past_key_values=drafter_cache,
                        use_cache=False,
                    )
                drafter_cache = Mq.past_key_values
                draft_logits = Mq.logits[..., -1, :]
                q[0, k] = draft_logits.to(self.device)
                xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
                input_ids = torch.cat((input_ids, xi), dim=1).to(self.device)

                draft_output[k] = self.tokenizer.decode(xi[0], skip_special_tokens=True)
                self.update_token(k, draft_output[k])           
                draft_step += 1

            token_request = protos.model_service_pb2.VerifyTokensRequest(
                user_uuid=self.generated_uuid
            )
            token_response = self.stub.VerifyTokens(token_request)

            if token_response.finished:
                break

            passed_tokens = token_response.passed_tokens
            verified_tokens = self.tokenizer.encode(token_response.verified_tokens, return_tensors='pt').to(self.device)
            if passed_tokens < self.generate_step:
                drafter_cache = drafter_cache[:-self.generate_step + passed_tokens]
                #input_ids = verified_tokens
                input_ids = torch.cat((input_ids[0][:-self.generate_step + passed_tokens].unsqueeze(0), verified_tokens), dim=1).to(self.device)
            else:
                #input_ids = verified_tokens
                input_ids = torch.cat((input_ids, verified_tokens), dim=1).to(self.device)

        output_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        end_time = time.time()
        print(f"Speculative 生成过程执行时间：{end_time - start_time} 秒")
        return output_text

    def traditional_generate(self, prompt, max_length=None):
        """
        使用传统方法生成文本。
        """
        start_time = time.time()
        request = protos.model_service_pb2.GenerateContentRequest(
            user_uuid=self.generated_uuid, 
            prompt=prompt, 
            max_length=self.max_length if max_length is None else max_length
        )

        try:
            response = self.stub.GenerateContent(request)
            end_time = time.time()
            print(f"传统生成过程执行时间：{end_time - start_time} 秒")
            return response.generated_text
        except grpc.RpcError as e:
            print(f"gRPC 错误: {e.code()} - {e.details()}")
            return None

    def compare_generate(self, prompt):
        """
        对比 speculative 和传统生成方法，输出两者的结果和时间。
        """
        print("开始 speculative 生成:")
        speculative_text = self.speculative_decode(prompt)
        
        print("\n开始传统生成:")
        traditional_text = self.traditional_generate(prompt)

        print("\nSpeculative 生成的文本:")
        print(speculative_text)

        print("\n传统生成的文本:")
        print(traditional_text)