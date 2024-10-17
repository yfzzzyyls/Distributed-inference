import grpc
from concurrent import futures
import protos.model_service_pb2
import protos.model_service_pb2_grpc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import traceback
import argparse
import threading
from collections import defaultdict
import json
import os

class UserConfigManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.user_configs = {}  # 存储每个用户的配置

    def get_user_config(self, user_uuid):
        with self.lock:
            return self.user_configs.get(user_uuid, None)

    def set_user_config(self, user_uuid, config):
        with self.lock:
            self.user_configs[user_uuid] = config


class ModelServiceServicer(protos.model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self, model_path: str, quantize: bool = False):
        """
        Initialize the model service with the specified model and quantization option.

        Args:
            model_path (str): The path to the model.
            quantize (bool): If True, apply quantization to the model weights.
            generate_step (int): The number of tokens to generate in each speculative iteration.
            max_length (int): The maximum length of the generated content.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if quantize:
            target_quantize = QuantoConfig(weights="int4")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=target_quantize)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.target_cache = None
        self.model.to(self.device)
        self.model.eval()
        self.config_manager = UserConfigManager()
        self.STORAGE_DIR = './user_data'
        self.user_data = defaultdict(lambda: {"tokens": "", "drafts": None, "logits": None})

    # 从文件加载用户数据（如果存在）
    def load_user_data(self, uuid):
        file_path = os.path.join(self.STORAGE_DIR, f"{uuid}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {"tokens": [], "logits": []}

    # 保存用户数据到文件
    def save_user_data(self, uuid, data):
        file_path = os.path.join(self.STORAGE_DIR, f"{uuid}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def PrepareSpeculative(self, request: protos.model_service_pb2.PrepareSpeculativeRequest, context: grpc.ServicerContext) -> protos.model_service_pb2.PrepareSpeculativeResponse:
        """
        Prepares speculative tokens based on the input text.

        Args:
            request (PrepareSpeculativeRequest): The gRPC request containing input text.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            PrepareSpeculativeResponse: A response containing the first generated tokens.
        """
        self.target_cache = None
        user_uuid = request.user_uuid
        prompt_text = request.prompt
        max_length = request.max_length
        generate_step = request.generate_step
        exact_mode = request.exact_mode
        debug_mode = request.debug_mode

        # Generate the first tokens
        first_tokens = self.prepare_speculative(user_uuid, prompt_text, max_length, generate_step, exact_mode, debug_mode)

        return protos.model_service_pb2.PrepareSpeculativeResponse(first_tokens=first_tokens)

    def prepare_speculative(self, user_uuid, prompt_text, max_length, generate_step, exact_mode, debug_mode) -> str:
        """
        Generate the first tokens based on the input text.

        Args:
            input_text (str): The input text for token generation.

        Returns:
            str: The generated text based on the input.
        """
        torch.cuda.empty_cache()
        self.config_manager.set_user_config(user_uuid, (max_length, generate_step, exact_mode, debug_mode))
        device = self.device
        input_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(device)

        # Use the model to generate output and update the cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self.target_cache,
                use_cache=False
            )
        self.target_cache = outputs.past_key_values

        # Get the next token using argmax
        next_token_logits = outputs.logits[..., -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        input_ids = torch.cat((input_ids, next_token_id), dim=1).to(device)

        # Convert tensor to text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.user_data[user_uuid]["tokens"] = generated_text
        self.user_data[user_uuid]["drafts"] = [None] * generate_step
        self.user_data[user_uuid]["logits"] = torch.zeros((1, generate_step, self.model.config.vocab_size), device=device)
        return generated_text
    
    def UpdateToken(self, request: protos.model_service_pb2.UpdateTokenRequest, context: grpc.ServicerContext) -> protos.model_service_pb2.UpdateTokenResponse:
        """
        Verifies the generated tokens against the input text and provided logits.

        Args:
            request (VerifyTokensRequest): The gRPC request containing input text and logits.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            TokenResponse: A response containing the verified tokens.
        """
        user_uuid = request.user_uuid
        k = request.index
        input_text = request.input_text
        generated_logits = request.generated_logits

        # Convert logits from request to tensor
        logits_list = []
        for float_array in generated_logits.rows:
            logits_list.append(float_array.values)

        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)
        # print(f"Received input text: {input_text}")
        # print(f"Received logits tensor: {logits_tensor.shape}")

        # Verify the tokens
        update_success = self.update_token(user_uuid, k, input_text, logits_tensor)

        return protos.model_service_pb2.UpdateTokenResponse(updated=update_success)

    def update_token(self, user_uuid, k, new_token: str, new_logits: torch.Tensor) -> str:
        self.user_data[user_uuid]["drafts"][k] = new_token
        self.user_data[user_uuid]["logits"][0, k] = new_logits.to(self.device)
        return True

    def VerifyTokens(self, request: protos.model_service_pb2.VerifyTokensRequest, context: grpc.ServicerContext) -> protos.model_service_pb2.VerifyTokensResponse:
        """
        Verifies the generated tokens against the input text and provided logits.

        Args:
            request (VerifyTokensRequest): The gRPC request containing input text and logits.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            TokenResponse: A response containing the verified tokens.
        """
        user_uuid = request.user_uuid

        # Verify the tokens
        verified_tokens = self.verify_tokens(user_uuid)

        return protos.model_service_pb2.VerifyTokensResponse(verified_tokens=verified_tokens)

    def verify_tokens(self, user_uuid: str) -> str:
        """
        Verifies the generated tokens using a rejection sampling method.

        Args:
            input_text (str): The input text.
            generated_tokens (torch.Tensor): The logits tensor for the generated tokens.

        Returns:
            str: The verified tokens as decoded text.
        """
        try:
            max_length, generate_step, exact_mode, debug_mode = self.config_manager.get_user_config(user_uuid)
            input_text = self.user_data[user_uuid]["tokens"] + "".join(self.user_data[user_uuid]["drafts"])
            if debug_mode:
                # print("logits shape: ", len(self.user_data[user_uuid]["logits"]))
                print(f"input_text: {input_text}")
            q = self.user_data[user_uuid]["logits"]
            
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

            Mp = self.model(
                input_ids=input_ids,
                past_key_values=self.target_cache, 
                use_cache=False,
            ) 
            self.target_cache = Mp.past_key_values
            draft_logits = Mp.logits[..., -generate_step-1: -1, :]
            p = draft_logits

            x = torch.argmax(p, dim=-1).unsqueeze(-1)
            target_tokens = x.squeeze().tolist()

            if isinstance(target_tokens, int):
                target_tokens = [target_tokens]
            
            n = generate_step
            for i, token in enumerate(target_tokens):
                if token != input_ids[0, -generate_step + i - 1]:
                    n = i

            '''if not exact_mode:
                r = torch.rand(generate_step, device=self.device)
            else:
                r = torch.ones(generate_step, device=self.device) 

            fractions = p / q
            n = generate_step
            for i in range(generate_step):
                if r[i] >= fractions[0, i, input_ids[0, -generate_step + i]]:
                    n = i
                    break'''

            p_p = Mp.logits[..., -generate_step + n - 1, :]
            x = torch.argmax(p_p, dim=-1).unsqueeze(-1)
            verified_tokens = x.squeeze().tolist()

            if isinstance(verified_tokens, int):
                verified_tokens = [verified_tokens]

            print(f"verified_tokens length: {n}")

            input_ids_flat = input_ids.squeeze(0).tolist()
            
            if n < generate_step:
                combined_tokens = input_ids_flat[:-generate_step + n] + verified_tokens
            else:
                combined_tokens = input_ids_flat + verified_tokens

            decoded_text = self.tokenizer.decode(combined_tokens, skip_special_tokens=True)

            
            if debug_mode:
                print(f"verified_tokens: {n}")
                print(f"verified_tokens: {decoded_text}")

            self.user_data[user_uuid]["tokens"] = decoded_text
            self.user_data[user_uuid]["logits"] = torch.zeros((1, generate_step, self.model.config.vocab_size), device=self.device)
            return decoded_text

        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred in verify_tokens: {e}")
            return ""

    def GenerateContent(self, request: protos.model_service_pb2.GenerateContentRequest, context: grpc.ServicerContext) -> protos.model_service_pb2.GenerateContentResponse:
        """
        Generate content based on a given prompt.

        Args:
            request (GenerateContentRequest): The gRPC request containing a prompt.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            GenerateContentResponse: A response containing the generated content.
        """
        user_uuid = request.user_uuid
        prompt = request.prompt
        self.target_cache = None
        max_length = request.max_length

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            generated_token_ids = []

            while len(input_ids[0]) < max_length:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=self.target_cache,
                        use_cache=False
                    )

                    self.target_cache = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1)

                    generated_token_ids.append(next_token_id.item())
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            all_tokens = self.tokenizer.encode(prompt) + generated_token_ids
            decoded_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
            torch.cuda.empty_cache()

            return protos.model_service_pb2.GenerateContentResponse(generated_text=decoded_text)
        except Exception as e:
            context.set_details(f"An error occurred: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            torch.cuda.empty_cache()
            return protos.model_service_pb2.GenerateContentResponse(generated_text="")


def serve(port: int, model_path: str, quantize: bool) -> None:
    """
    Start the gRPC server to serve model-related requests.

    Args:
        port (int): The port to listen on.
        model_path (str): The path to the model.
        quantize (bool): Whether to apply quantization to the model.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    try:
        protos.model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            ModelServiceServicer(model_path, quantize), server)
        server.add_insecure_port(f'[::]:{port}')
        server.start()
        print(f"服务器已启动，监听端口 {port}")
        server.wait_for_termination()
    except Exception as e:
        print(f"Failed to start the server on port {port}: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='启动模型服务')
    parser.add_argument('--port', type=int, default=50051, help='指定监听的端口')
    parser.add_argument('--model-path', type=str, default="/home/apc/llama/Llama-3.1-8B-Instruct", help='指定模型路径')
    parser.add_argument('--quantize', action='store_true', help='是否启用量化')
    args = parser.parse_args()

    serve(args.port, args.model_path, args.quantize)
