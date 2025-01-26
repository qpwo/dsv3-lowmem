import os
import json
from argparse import ArgumentParser
from typing import List, Union

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from model import Linear, Transformer, ModelArgs
import time
realprint = print

print0 = realprint

torch.set_num_threads(32)
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

rank = int(os.getenv("RANK", "0"))

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    tokenizer = None
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    started_at = time.time()
    def print_rate():
        elapsed = time.time() - started_at
        print0(f"\nDid {numdid} tokens in {elapsed:.2f} seconds ({numdid / elapsed:.1f} tok/sec)\n")

    numdid = 0
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # if tokenizer and rank == 0:
        #     string = tokenizer.decode(tokens[0, cur_pos:cur_pos+1].tolist(), skip_special_tokens=True)
        #     realprint(string, flush=True, end='')
        if rank == 0:
            realprint('.', flush=True, end='')
            if numdid % 10 == 0:
                print_rate()

        numdid += 1

        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    print_rate()
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


from datetime import datetime
import pytz

pacific_tz = pytz.timezone('America/Los_Angeles')
def stamp():
    t = datetime.now(pacific_tz)
    return t.strftime("%I:%M:%S %p")

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    def print(*args, **kwargs):
        realprint(f"{stamp()} [gpu_{rank}]", *args, **kwargs)
    global print0
    def print0(*args, **kwargs):
        if rank != 0:
            return
        realprint(f"{stamp()} [gpu_{rank}]", *args, **kwargs)
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    print('making model')
    with torch.device("cuda"):
        model = Transformer(args)
    print('loading model')
    weight_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    my_load_model(model, weight_path)
    print('fixing weight.scale')
    for module in model.modules():  # modules() iterates through all descendants including self
        if isinstance(module, Linear):
            module.weight.scale = module.scale
    print('firing up')
    # time.sleep(3)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])

    if interactive:
        messages = []
        counter = -1
        while True:
            counter += 1
            def inpoot():
                # if counter == 0:
                if counter < 2:
                    return "WARM ME UP. TELL ME A LONG STORY. LET'S GET WARM."
                return input(">>> ")
            if world_size == 1:
                prompt = inpoot()
            elif rank == 0:
                prompt = inpoot()
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, tokenizer=tokenizer)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print0('\n' + completion + '\n')
            messages.append({"role": "assistant", "content": completion})

            messages.clear() # comment me to enable history
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print0("Prompt:", prompt)
            print0("Completion:", completion)
            print0()

    if world_size > 1:
        dist.destroy_process_group()


import torch.multiprocessing

def my_load_model(
    model: torch.nn.Module, filename: Union[str, os.PathLike]
):
    filename = str(filename)
    import safetensors.torch
    total = torch.tensor(0., device='cpu')
    print(f"loading {filename}")
    sd = safetensors.torch.load_file(filename, device="cpu")
    for k in list(sd.keys()):
        if '.experts.' not in k:
            sd[k] = sd[k].to('cuda')
        total += sd[k].view(-1)[0].float().cpu() # force it to actually load
    model.load_state_dict(sd, strict=False, assign=True)

if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
