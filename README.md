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
