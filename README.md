Run deepseek v3 on a 8xh100 node! Or 6 gh200s.

### 6.1 Inference with DeepSeek-Infer Demo (example only)

#### System Requirements

Dependencies:
```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```
#### Model Weights & Demo Code Preparation

First, clone our DeepSeek-V3 GitHub repository:

```shell
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```

Navigate to the `inference` folder and install dependencies listed in `requirements.txt`. Easiest way is to use a package manager like `conda` or `uv` to create a new virtual environment and install the dependencies.

```shell
cd DeepSeek-V3/inference
pip install -r requirements.txt
```

Download the model weights from Hugging Face, and put them into `/path/to/DeepSeek-V3` folder.

#### Model Weights Conversion

Convert Hugging Face model weights to a specific format:

```shell
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

#### Run

Then you can chat with DeepSeek-V3:

```shell
torchrun --nnodes 1 --nproc-per-node 8 generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
```

Or batch inference on a given file:

```shell
torchrun --nnodes 1 --nproc-per-node 8 generate.py --ckpt-path /path/to/DeepSeek-V3-Demo --config configs/config_671B.json --input-file $FILE
```
