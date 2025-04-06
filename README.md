# Distributed Inference with Speculative Decoding on AWS Trainium

This repository has been adapted for **single-device** AWS Trainium usage with **speculative decoding** by default, using **Meta LLaMA 3.2** (1B draft + 3B target) in **bfloat16**. We assume you have an **AWS DLAMI** with Neuron SDK installed.

## Dpendencies

install dependencies

```
pip install grpcio==1.71.0 grpcio-tools==1.66.2
pip install gevent
```

## Setup

1. **Clone Repo & Install**:

   ```bash
   git clone https://github.com/yfzzzyyls/Distributed-inference.git
   cd Distributed-inference
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```
3. **(Optional) Compile Offline**
   You can compile each model to a TorchScript file once, so future runs skip JIT:

```bash
# Target model
python model_service.py --model-path /home/ubuntu/models/llama-3.2-1b --compiled-model-path /home/ubuntu/models/llama1b_neuron.pt --compile --max-context 128
# Draft model
python draft_client.py --model /home/ubuntu/models/llama-3.2-1b/ --compiled_model_path /home/ubuntu/models/llama1b_neuron_draft.pt --compile_model --max_length 32 --gamma 4 --prompt "Once upon a time,"
```

**This will produce **models/llama3b_neuron.pt** and **models/llama1b_neuron.pt**.**

4. generate new grpc files

   ```
   cd ~/Distributed-inference/protos
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. model_service.proto
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. service.proto
   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. batch.proto
   ```

   Notice: if you encounter import failure issue:

   replace:

   ```
   import model_service_pb2 as model__service__pb2
   ```

   to:

   ```
   from . import model_service_pb2 as model__service__pb2
   ```

## **Usage: Single-Device Trainium**

### **Start the Target Model Server**

```
python model_service.py --model-path /home/ubuntu/models/llama-3.2-1b --compiled-model-path /home/ubuntu/models/llama1b_neuron.pt --compile --max-context 128
```

* This loads or compiles the 3B model for the Trainium device.
* The server listens on port 50051 for gRPC calls.
* Adjust **--max-context** if you want a different maximum context length for KV cache.

### **Start the Draft Client**

In another terminal on the same instance:

```
python draft_client.py --host localhost --port 50051 --model /home/ubuntu/models/llama-3.2-1b/ --compiled_model_path /home/ubuntu/models/llama1b_neuron_draft.pt --max_length 32 --gamma 4 --prompt "Once upon a time,"
```

* The client loads or compiles the 1B draft model.
* **--gamma 4** means we generate 4 tokens at a time speculatively.
* By default, we do incremental decoding with KV cache. The client sends tokens to the server for verification.
* The final output text is printed with timing/throughput info.
