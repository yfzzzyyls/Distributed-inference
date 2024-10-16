import grpc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig
import model_service_pb2
import model_service_pb2_grpc
import time
import argparse
from datasets import load_dataset
import matplotlib.pyplot as plt


def load_model_and_tokenizer(model_name, quantize_config):
    """
    加载模型和 tokenizer，并返回模型、tokenizer 和词汇表大小。
    """
    tokenizer = AutoTokenizer.from_pretrained("/home/apc/llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantize_config)
    model.eval()  # 设置为评估模式
    vocabulary_size = model.config.vocab_size

    return tokenizer, model, vocabulary_size

def grpc_client(stub, prompt, model, tokenizer, device, vocabulary_size, max_length, generate_step):
    """
    与 gRPC 服务进行交互，执行 token 验证和生成任务（speculative 模式）。
    """
    drafter_cache = None
    total_generated = 0  # 已生成的 tokens 数
    first_target = True
    output_text = ""
    
    start_time = time.time()  # 记录开始时间

    if first_target:
        # 运行目标模型获取第一个 token
        prepare_request = model_service_pb2.PrepareSpeculativeRequest(input_text=prompt)
        prepare_response = stub.PrepareSpeculative(prepare_request)
        first_tokens = prepare_response.first_tokens  # 这是一个 token ID 列表
        output_text = first_tokens
        total_generated += 1

    while total_generated < max_length:
        q = torch.zeros((1, generate_step, vocabulary_size), device=device)
        input_ids = tokenizer.encode(output_text, return_tensors='pt').to(device)
        total_generated = len(input_ids[0])
        if total_generated >= max_length:
            break

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

        draft_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        logits_to_send = q.detach().cpu().numpy().tolist()  # 转换为列表

        # 构建 FloatArray 列表
        float_array_list = [model_service_pb2.FloatArray(values=row) for row in logits_to_send[0]]

        # 构建 FloatArray2D 对象
        float_array_2d = model_service_pb2.FloatArray2D(rows=float_array_list)

        # 发送生成的 tokens 进行验证
        token_request = model_service_pb2.VerifyTokensRequest(
            input_text=draft_output,
            generated_logits=float_array_2d
        )
        token_response = stub.VerifyTokens(token_request)

        # 更新输出文本为验证通过的 tokens
        output_text = token_response.verified_tokens

    output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    end_time = time.time()  # 记录结束时间
    print(f"Speculative 生成过程执行时间：{end_time - start_time} 秒")
    torch.cuda.empty_cache()

    return output_text, end_time - start_time

def traditional_generate(stub, prompt):
    """
    使用传统方法生成文本。
    """
    start_time = time.time()
    request = model_service_pb2.GenerateContentRequest(prompt=prompt)

    # 调用 GenerateContent 方法
    try:
        response = stub.GenerateContent(request)
        end_time = time.time()  # 记录开始时间
        print(f"传统生成过程执行时间：{end_time - start_time} 秒")
    except grpc.RpcError as e:
        print(f"gRPC 错误: {e.code()} - {e.details()}")

    return response.generated_text, end_time - start_time

def compare_generate(stub, prompt, model, tokenizer, device, max_length, generate_step, time_list):
    """
    对比 speculative 和传统生成方法，输出两者的结果和时间。
    """
    print("开始 speculative 生成:")
    speculative_text, draft_time = grpc_client(stub, prompt, model, tokenizer, device, model.config.vocab_size, max_length, generate_step)
    
    print("\nSpeculative 生成的文本:")
    print(speculative_text)

    print("\n开始传统生成:")
    traditional_text, target_time = traditional_generate(stub, prompt)

    print("\n传统生成的文本:")
    print(traditional_text)

    time_list.append((draft_time, target_time, target_time // draft_time))
    print(f"\n加速比：{target_time / draft_time * 100}")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="gRPC 模型生成客户端")
    parser.add_argument('--model', type=str, default='/home/apc/llama/Llama-3.2-1B-Instruct', help='模型路径或名称')
    parser.add_argument('--quantize', action='store_true', help='是否启用量化')
    parser.add_argument('--max_length', type=int, default=50, help='最大生成长度')
    parser.add_argument('--generate_step', type=int, default=6, help='每次生成的 tokens 数')
    parser.add_argument('--method', type=str, choices=['speculative', 'traditional', 'compare'], default='speculative', help='选择生成方法')
    args = parser.parse_args()

    # 配置
    MODEL_NAME = args.model  # 模型路径
    SERVER_ADDRESS = 'localhost:3000'  # 服务器地址
    draft_quantize = QuantoConfig(weights="int4") if args.quantize else None
    MAX_LENGTH = args.max_length  # 最大生成长度
    GENERATE_STEP = args.generate_step  # 每次生成的 tokens 数

    # 加载模型和 tokenizer
    tokenizer, model, vocabulary_size = load_model_and_tokenizer(MODEL_NAME, draft_quantize)

    # 检查是否可用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化 gRPC 通道和存根
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    time_list = []

    dataset = load_dataset("xsum", split="test[:5]")  # 这里只用前10个样本进行演示

    # 循环接受多次输入
    for i, example in enumerate(dataset):
        prompt = example["document"]
        print(f"第 {i + 1} 个样本：{prompt}")
        
        compare_generate(stub, prompt, model, tokenizer, device, MAX_LENGTH, GENERATE_STEP, time_list)

    # 分别从time_list中提取draft_time, target_time和比值
    draft_times = [item[0] for item in time_list]
    target_times = [item[1] for item in time_list]
    ratios = [item[2] for item in time_list]

    # 创建X轴的标签（这里使用每个数据点的索引作为x轴）
    x_labels = range(1, len(time_list) + 1)

    # 绘制draft_time和target_time的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, draft_times, label="Draft Time", marker='o')
    plt.plot(x_labels, target_times, label="Target Time", marker='s')

    # 绘制比值的柱状图
    plt.bar(x_labels, ratios, alpha=0.3, label="Ratio (Target/Draft)", color='green')

    # 添加图例、标签和标题
    plt.xlabel("Test Case Index")
    plt.ylabel("Time (Milliseconds)")
    plt.title("Llama-3.2-1B-Instruct & Llama-3.1-8B XSum Test Time Analysis (Exact Mode)")
    plt.legend()

    # 展示图表
    plt.savefig('xsum_time_analysis_exact_plot.pdf')

    time_list = []

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")  # 使用测试集中前10个样本作为演示

    # 循环接受多次输入
    for i, example in enumerate(dataset):
        prompt = example["article"]
        print(f"第 {i + 1} 个样本：{prompt}")
        
        compare_generate(stub, prompt, model, tokenizer, device, MAX_LENGTH, GENERATE_STEP, time_list)

    # 分别从time_list中提取draft_time, target_time和比值
    draft_times = [item[0] for item in time_list]
    target_times = [item[1] for item in time_list]
    ratios = [item[2] for item in time_list]

    # 创建X轴的标签（这里使用每个数据点的索引作为x轴）
    x_labels = range(1, len(time_list) + 1)

    # 绘制draft_time和target_time的折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, draft_times, label="Draft Time", marker='o')
    plt.plot(x_labels, target_times, label="Target Time", marker='s')

    # 绘制比值的柱状图
    plt.bar(x_labels, ratios, alpha=0.3, label="Ratio (Target/Draft)", color='green')

    # 添加图例、标签和标题
    plt.xlabel("Test Case Index")
    plt.ylabel("Time (Milliseconds)")
    plt.title("Llama-3.2-1B-Instruct & Llama-3.1-8B CNNDM Test Time Analysis (Exact Mode)")
    plt.legend()

    # 展示图表
    plt.savefig('cnndm_time_analysis_exact_plot.pdf')

if __name__ == '__main__':
    main()
