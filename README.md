Run deepseek v3 on a 8xh100 node! Or 6 gh200s.

This works by offloading unused "experts" (as in MoE) to CPU.

You can change `min_free_mb` in `model.py`. I have it set to 2 GB.

python 3.10 is recommended

```
# you can copy-paste this whole code block
pip install hf_transfer 'huggingface_hub[hf_transfer]'
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --max-workers=32 --local-dir ~/dsv3 deepseek-ai/DeepSeek-V3

git clone https://github.com/qpwo/dsv3-lowmem
cd dsv3-lowmem/inference
pip install -r requirements.txt
python convert.py --hf-ckpt-path ~/dsv3 --save-path ~/dsv3-mp8 --n-experts 256 --model-parallel 8
OMP_NUM_THREADS=64 torchrun --nnodes 1 --nproc-per-node 8 generate.py --ckpt-path ~/dsv3-mp8 --config configs/config_671B.json --interactive --temperature 0.7 --max-new-tokens 200
# ^ First two responses will be slow. Fast after that.
```

Or batch inference on a given file:

```shell
OMP_NUM_THREADS=64 torchrun --nnodes 1 --nproc-per-node 8 generate.py --ckpt-path ~/dsv3-mp8 --config configs/config_671B.json --input-file $FILE
```
