import grpc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import protos.model_service_pb2
import protos.model_service_pb2_grpc
import time
import argparse
import uuid
import asyncio


def load_model_and_tokenizer(model_name, quantize_config):
    """
    加载模型和 tokenizer，并返回模型、tokenizer 和词汇表大小。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantize_config)
    model.eval()  # 设置为评估模式
    vocabulary_size = model.config.vocab_size
    return tokenizer, model, vocabulary_size

def update_token(stub, uuid, k, draft_output, logits_to_send):
     # 构建 FloatArray 列表
    float_array_list = [protos.model_service_pb2.FloatArray(values=row) for row in logits_to_send]

    # 构建 FloatArray2D 对象
    float_array_2d = protos.model_service_pb2.FloatArray2D(rows=float_array_list)

    token_update_request = protos.model_service_pb2.UpdateTokenRequest(
        user_uuid=uuid,
        index = k,
        input_text=draft_output,
        generated_logits=float_array_2d
    )

    return stub.UpdateToken(token_update_request)

async def process_step(queue, step_index, stub, uuid, draft_output, logits_to_send):
    loop = asyncio.get_running_loop()
    
    # 将计算和同步的 update_token 调用放入异步任务
    result = await loop.run_in_executor(
        None,  # 使用默认的线程池执行器
        update_token,  # 调用的同步函数
        stub, uuid, step_index, draft_output, logits_to_send  # 参数
    )
    
    # 将结果按顺序放入队列中
    await queue.put((step_index, result))

def grpc_client(stub, uuid, prompt, model, tokenizer, device, vocabulary_size, max_length, generate_step):
    """
    与 gRPC 服务进行交互，执行 token 验证和生成任务（speculative 模式）。
    """
    drafter_cache = None
    debug_mode = True
    total_generated = 0  # 已生成的 tokens 数
    first_target = True
    output_text = ""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    
    start_time = time.time()  # 记录开始时间

    if first_target:
        # 运行目标模型获取第一个 token
        if debug_mode:
            print(type(uuid), type(prompt), type(max_length), type(generate_step))

        prepare_request = protos.model_service_pb2.PrepareSpeculativeRequest(user_uuid = uuid, prompt=prompt, max_length=max_length, generate_step=generate_step, exact_mode=False, debug_mode=False)
        prepare_response = stub.PrepareSpeculative(prepare_request)
        first_tokens = prepare_response.first_tokens  # 这是一个 token ID 列表
        # print(f"第一个 token: {first_tokens}")
        output_text = first_tokens
        total_generated += 1

    # print("max_length: ", max_length)
    while total_generated < max_length:
        q = torch.zeros((1, generate_step, vocabulary_size), device=device)
        input_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)
        total_generated = len(input_ids[0])
        if total_generated >= max_length:
            # print("已达到最大生成长度")
            break
        
        tasks = []
        for k in range(generate_step):
            with torch.no_grad():
                Mq = model(
                    input_ids=input_ids,
                    past_key_values=drafter_cache,
                    use_cache=False,
                )
            drafter_cache = Mq.past_key_values
            draft_logits = Mq.logits[..., -1, :]
            q[0, k] = draft_logits.to(device)
            xi = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat((input_ids, xi), dim=1).to(device)

            draft_output = tokenizer.decode(xi[0], skip_special_tokens=True)

            logits_to_send = draft_logits.detach().cpu().numpy().tolist()  # 转换为列表

            # print(f"已生成的 tokens: {draft_output}")
            task = loop.create_task(process_step(queue, k, stub, uuid, draft_output, logits_to_send))
            tasks.append(task)

        loop.run_until_complete(asyncio.gather(*tasks))
        # 发送生成的 tokens 进行验证
        token_request = protos.model_service_pb2.VerifyTokensRequest(
            user_uuid=uuid
        )
        token_response = stub.VerifyTokens(token_request)

        # 更新输出文本为验证通过的 tokens
        output_text = token_response.verified_tokens
        print(f"已验证的 tokens: {output_text}")

    # output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    end_time = time.time()  # 记录结束时间
    print(f"Speculative 生成过程执行时间：{end_time - start_time} 秒")

    return output_text

def traditional_generate(stub, uuid, prompt, max_length):
    """
    使用传统方法生成文本。
    """
    start_time = time.time()
    request = protos.model_service_pb2.GenerateContentRequest(user_uuid=uuid, prompt=prompt, max_length=max_length)

    # 调用 GenerateContent 方法
    try:
        response = stub.GenerateContent(request)
        end_time = time.time()  # 记录开始时间
        print(f"传统生成过程执行时间：{end_time - start_time} 秒")
        print(response.generated_text)
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")

    return response.generated_text

def compare_generate(stub, uuid, prompt, model, tokenizer, device, max_length, generate_step):
    """
    对比 speculative 和传统生成方法，输出两者的结果和时间。
    """
    print("开始 speculative 生成:")
    speculative_text = grpc_client(stub, uuid, prompt, model, tokenizer, device, model.config.vocab_size, max_length, generate_step)
    
    print("\n开始传统生成:")
    traditional_text = traditional_generate(stub, uuid, prompt, max_length)

    print("\nSpeculative 生成的文本:")
    print(speculative_text)

    print("\n传统生成的文本:")
    print(traditional_text)

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="gRPC 模型生成客户端")
    parser.add_argument('--model', type=str, default='/home/apc/llama/Llama-3.2-1B-Instruct', help='模型路径或名称')
    parser.add_argument('--quantize', action='store_true', help='是否启用量化')
    parser.add_argument('--debug', action='store_true', help='是否启用debug')
    parser.add_argument('--max_length', type=int, default=50, help='最大生成长度')
    parser.add_argument('--generate_step', type=int, default=6, help='每次生成的 tokens 数')
    parser.add_argument('--method', type=str, choices=['speculative', 'traditional', 'compare'], default='speculative', help='选择生成方法')
    parser.add_argument('--server_address', type=str, default='localhost:2024', help='服务器地址')    
    args = parser.parse_args()

    # 配置
    MODEL_NAME = args.model  # 模型路径
    SERVER_ADDRESS = args.server_address  # 服务器地址
    debug_mode = True if args.debug else False
    draft_quantize = QuantoConfig(weights="int4") if args.quantize else None
    max_length = args.max_length  # 最大生成长度
    generate_step = args.generate_step  # 每次生成的 tokens 数

    # 加载模型和 tokenizer
    tokenizer, model, vocabulary_size = load_model_and_tokenizer(MODEL_NAME, draft_quantize)

    # 检查是否可用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化 gRPC 通道和存根
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = protos.model_service_pb2_grpc.ModelServiceStub(channel)

    generated_uuid = str(uuid.uuid4())
    print(f"生成的 UUID: {generated_uuid}")

    # 循环接受多次输入
    while True:
        prompt = input("请输入文本提示（输入 'exit' 退出）：")
        if prompt.lower() == 'exit':
            print("退出程序")
            break

        # 根据选择的生成方法调用对应的逻辑
        if args.method == 'speculative':
            client_output = grpc_client(stub, generated_uuid, prompt, model, tokenizer, device, model.config.vocab_size, max_length, generate_step)
            print("\nSpeculative 生成的文本:")
            print(client_output)
        elif args.method == 'traditional':
            client_output = traditional_generate(stub, generated_uuid, prompt, max_length)
            print("\n传统生成的文本:")
            print(client_output)
        elif args.method == 'compare':
            compare_generate(stub, generated_uuid, prompt, model, tokenizer, device, max_length, generate_step)


if __name__ == '__main__':
    main()
