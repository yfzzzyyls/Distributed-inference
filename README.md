# MultiDeviceSpeculativeDecoding

**Author**: Yunhai Hu

## Project Overview

The **MultiDeviceSpeculativeDecoding** project aims to decouple the speculative decoding process by separating the draft and target models, allowing for flexible deployment across different devices. The key idea is to perform draft generation on the user's device, while the verification of the generated tokens is handled by a remote service. This approach enables efficient speculative decoding across distributed systems, optimizing both performance and scalability.

### Key Features

1. **Decoupled Speculative Decoding**: 
   - Draft generation is executed on the user's device.
   - Target model verification is performed by a remote service, enabling better load distribution and resource utilization.
   
2. **Multi-Device Flexibility**: 
   - Deploy the draft model and target model across different devices or machines to maximize flexibility and efficiency.
   
3. **Batch Processing (Future Support)**:
   - The system will support batch processing of user requests, optimizing throughput and latency for large-scale operations.
   
4. **Variable Length Requests (Future Support)**:
   - Handling requests of different lengths from various users and adapting the speculative decoding process to requests with variable lengths of drafts.

5. **Dynamic Speculative Draft Lengths (Future Support)**:
   - Processing drafts with different lengths dynamically, allowing for flexible and efficient speculative decoding.

## Architecture

The architecture of the project revolves around the separation of draft generation and verification, which are performed on different devices. The user device generates the speculative draft based on an initial input prompt, and the service performs verification and final decoding.

- **User Device (Draft Model)** :
  - Performs **Draft Generation** using a lightweight model.
  - Sends draft outputs to the service for verification.
  
- **Service (Verification Model)**:
  - Receives draft tokens from the user device.
  - Performs **Token Verification** using a more powerful target model.
  - Returns verified tokens to the user device for further processing.

## How to Run

### Requirements

- Python 3.8+
- [gRPC](https://grpc.io/) and [Protocol Buffers](https://developers.google.com/protocol-buffers) (for service communication)
- PyTorch
- [Transformers](https://huggingface.co/transformers) library from Hugging Face
- CUDA (if using GPU)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YunhaiHu/MultiDeviceSpeculativeDecoding.git
   cd MultiDeviceSpeculativeDecoding
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Compile the proto files for gRPC communication (ensure protoc is installed):
   ```bash
   python -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. model_service.proto

### Running the Service
1. Start the gRPC service to handle token verification
    ```bash
    python server.py --model "/path/to/target_model" --port 50051

### Running the Client
1. Start the client to generate drafts and interact with the service:
    ```bash
    python client.py --model "/path/to/draft_model" --max_length 100 --generate_step 8 --method speculative
2. To compare the speculative method with traditional generation:
    ```bash
    python client.py --model "/path/to/draft_model" --max_length 100 --generate_step 8 --method compare

## Future Development
**Batch Processing:** Support for handling batch requests from multiple users, enabling the system to process multiple speculative decoding tasks simultaneously.

**Variable Length Handling:** Efficiently handle requests from different users with varying input and output lengths, dynamically adjusting the speculative decoding length for each request.

**Optimized Deployment:** Investigate distributed deployment strategies to handle verification models across multiple machines, improving scalability and fault tolerance.

## Contribution
Contributions are welcome! Please feel free to open issues or submit pull requests to enhance the project.

