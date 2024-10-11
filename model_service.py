import grpc
from concurrent import futures
import model_service_pb2
import model_service_pb2_grpc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    QuantoConfig,
)
import traceback
import argparse


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self, model_path: str = "/home/apc/llama/Llama-3.1-8B-Instruct", quantize: bool = False, generate_step: int = 6, max_length: int = 50, exact_mode: bool = False):
        """
        Initialize the model service with the specified model and quantization option.

        Args:
            model_path (str): The path to the model.
            quantize (bool): If True, apply quantization to the model weights.
            generate_step (int): The number of tokens to generate in each speculative iteration.
            max_length (int): The maximum length of the generated content.
        """
        if quantize:
            target_quantize = QuantoConfig(weights="int4")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=target_quantize)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.target_cache = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.GENERATE_STEP = generate_step  # Number of tokens to generate at each step
        self.max_length = max_length  # Max length of content generation
        self.exact_mode = exact_mode

    def PrepareSpeculative(self, request: model_service_pb2.PrepareSpeculativeRequest, context: grpc.ServicerContext) -> model_service_pb2.PrepareSpeculativeResponse:
        """
        Prepares speculative tokens based on the input text.

        Args:
            request (PrepareSpeculativeRequest): The gRPC request containing input text.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            PrepareSpeculativeResponse: A response containing the first generated tokens.
        """
        self.target_cache = None
        input_text = request.input_text

        # Generate the first tokens
        first_tokens = self.prepare_speculative(input_text)

        return model_service_pb2.PrepareSpeculativeResponse(first_tokens=first_tokens)

    def prepare_speculative(self, input_text: str) -> str:
        """
        Generate the first tokens based on the input text.

        Args:
            input_text (str): The input text for token generation.

        Returns:
            str: The generated text based on the input.
        """
        device = self.device
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(device)

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
        return generated_text

    def VerifyTokens(self, request: model_service_pb2.VerifyTokensRequest, context: grpc.ServicerContext) -> model_service_pb2.VerifyTokensResponse:
        """
        Verifies the generated tokens against the input text and provided logits.

        Args:
            request (VerifyTokensRequest): The gRPC request containing input text and logits.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            TokenResponse: A response containing the verified tokens.
        """
        input_text = request.input_text
        generated_logits = request.generated_logits

        # Convert logits from request to tensor
        logits_list = []
        for float_array in generated_logits.rows:
            logits_list.append(float_array.values)

        logits_tensor = torch.tensor(logits_list, dtype=torch.float32).to(self.device)

        # Verify the tokens
        verified_tokens = self.verify_tokens(input_text, logits_tensor)

        return model_service_pb2.VerifyTokensResponse(verified_tokens=verified_tokens)

    def verify_tokens(self, input_text: str, generated_tokens: torch.Tensor) -> str:
        """
        Verifies the generated tokens using a rejection sampling method.

        Args:
            input_text (str): The input text.
            generated_tokens (torch.Tensor): The logits tensor for the generated tokens.

        Returns:
            str: The verified tokens as decoded text.
        """
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            q = torch.tensor(generated_tokens).unsqueeze(0).to(self.device)

            Mp = self.model(
                input_ids=input_ids,
                past_key_values=self.target_cache,
                use_cache=False,
            )
            self.target_cache = Mp.past_key_values

            draft_logits = Mp.logits[..., -self.GENERATE_STEP-1: -1, :]
            p = draft_logits

            if not self.exact_mode:
                r = torch.rand(self.GENERATE_STEP, device=self.device)
            else:
                r = torch.ones(self.GENERATE_STEP, device=self.device) 

            fractions = p / q
            n = self.GENERATE_STEP
            for i in range(self.GENERATE_STEP):
                if r[i] > fractions[0, i, input_ids[0, -self.GENERATE_STEP + i]]:
                    n = i
                    break

            p_p = Mp.logits[..., -self.GENERATE_STEP + n - 1, :]
            x = torch.argmax(p_p, dim=-1).unsqueeze(-1)
            verified_tokens = x.squeeze().tolist()

            if isinstance(verified_tokens, int):
                verified_tokens = [verified_tokens]

            input_ids_flat = input_ids.squeeze(0).tolist()
            
            if n < self.GENERATE_STEP:
                combined_tokens = input_ids_flat[:-self.GENERATE_STEP + n] + verified_tokens
            else:
                combined_tokens = input_ids_flat + verified_tokens

            decoded_text = self.tokenizer.decode(combined_tokens, skip_special_tokens=True)

            return decoded_text

        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred in verify_tokens: {e}")
            return ""

    def GenerateContent(self, request: model_service_pb2.GenerateContentRequest, context: grpc.ServicerContext) -> model_service_pb2.GenerateContentResponse:
        """
        Generate content based on a given prompt.

        Args:
            request (GenerateContentRequest): The gRPC request containing a prompt.
            context (grpc.ServicerContext): The gRPC context.

        Returns:
            GenerateContentResponse: A response containing the generated content.
        """
        prompt = request.prompt
        self.target_cache = None

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            generated_token_ids = []

            for _ in range(self.max_length):
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

            return model_service_pb2.GenerateContentResponse(generated_text=decoded_text)
        except Exception as e:
            context.set_details(f"An error occurred: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return model_service_pb2.GenerateContentResponse(generated_text="")


def serve(port: int, model_path: str, quantize: bool, generate_step: int, max_length: int, exact_mode: bool) -> None:
    """
    Start the gRPC server to serve model-related requests.

    Args:
        port (int): The port to listen on.
        model_path (str): The path to the model.
        quantize (bool): Whether to apply quantization to the model.
        generate_step (int): Number of tokens generated in speculative requests.
        max_length (int): Maximum length of generated content.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    try:
        model_service_pb2_grpc.add_ModelServiceServicer_to_server(
            ModelServiceServicer(model_path, quantize, generate_step, max_length, exact_mode), server)
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
    parser.add_argument('--generate-step', type=int, default=6, help='每次生成的 token 数量')
    parser.add_argument('--max-length', type=int, default=50, help='生成内容的最大长度')
    parser.add_argument('--exact_mode', action='store_true', help='是否启用精准模式')
    args = parser.parse_args()

    serve(args.port, args.model_path, args.quantize, args.generate_step, args.max_length, args.exact_mode)
