# Distributed Inference with Speculative Decoding on AWS Trainium

This repository has been adapted for **single-device** AWS Trainium usage with **speculative decoding** by default, using **Meta LLaMA 3.2** (1B draft + 3B target) in **bfloat16**. We assume you have an **AWS DLAMI** with Neuron SDK installed.

## Setup

1. **Clone Repo & Install**:

   ```bash
   git clone https://github.com/yfzzzyyls/Distributed-inference.git
   cd Distributed-inference
   pip install -r requirements.txt
   # Ensure torch-neuronx, transformers[neuron], grpcio, etc. are installed
   ```
2. **Download Models** (1B draft, 3B target) from Hugging Face. For example:

   ```
   cd ~
   mkdir models
   huggingface-cli download --token YOURTOKEN meta-llama/Llama-3.2-1B --local-dir /home/ubuntu/models/llama-3.2-1b
   ```

from transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained(‘meta-llama/Llama-3.2-1B-Instruct’, torch_dtype=‘auto’).save_pretrained(‘models/llama-3.2-1b’)

AutoModelForCausalLM.from_pretrained(‘meta-llama/Llama-3.2-3B-Instruct’, torch_dtype=‘auto’).save_pretrained(‘models/llama-3.2-3b’)

Each folder should contain `pytorch_model.bin`, `config.json`, `tokenizer.model` or merges/vocab, etc.

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

### **Example Output**

```
[Draft] Final output: Once upon a time, ...
[Draft] Time: 2.35 s, tokens: 20, speed: 8.51 tokens/s
```

## **Performance Testing**

Run the **evaluate_test.py** script to compare speculative decoding vs. target-only:

1. **Ensure server is running** with the 3B model.
2. Launch:

```
python evaluate_test.py
  --host localhost --port 50051
  --draft_model models/llama-3.2-1b
  --compiled_draft_model models/llama1b_neuron.pt
  --compile
  --max_tokens 128 --gamma 4
  --prompt "Once upon a time,"
  --target_model_service
```

You’ll see something like:

```
Speculative decoding result:
Once upon a time, ...
Spec time: 2.12s, tokens=40, throughput=18.87 t/sBaseline target-only result:
Once upon a time, ...
Baseline time: 3.95s, tokens=40, throughput=10.12 t/s
```

This shows ~1.8x speedup from speculative decoding.

## **Advanced Tips**

* **NEURON_RT_VISIBLE_CORES**: If your instance has multiple NeuronCores, you can dedicate certain cores to the draft or server processes:

```
#In terminal 1 (server):export NEURON_RT_VISIBLE_CORES=4-15
python model_service.py ...#In terminal 2 (draft):export NEURON_RT_VISIBLE_CORES=0-3
python draft_client.py ...
```

This can allow parallel execution, improving throughput.

* **Larger Models**: If using LLaMA 7B or bigger, you might need to distribute the model across multiple Neuron cores. That requires advanced compilation with **neuronx-distributed** or optimum-neuron. The approach is similar; just ensure the code references the sharded model.
* **Modifying the Speculative Mechanism**: The draft code uses a simple loop with **use_cache=True**. If you want to do partial or multi-token steps differently, you can adapt the logic in **draft_client.py**
