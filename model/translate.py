from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from nltk.translate.bleu_score import sentence_bleu
import jieba
import llama
import argparse
from accelerate.utils import set_seed
import json
import tensor_parallel as tp

import os
from tqdm import tqdm

PROMPT_DICT = {
    # "prompt_instruct": (
    #     "以下是一个描述任务的指令，并配有一个提供详细上下文信息的输入。"
    #     "请写一个完成该指令的适当回复。\n\n"
    #     "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回复:"
    # ),
    "prompt_input": (
        "以下是一个描述任务的指令，请写一个完成该指令的适当回复。\n\n"
        "### 指令:\n{0}\n\n### 回复:"
    ),
    "translate_prompt":"{0}句子：“{2}”的{1}是：",
    "translate_instruct": "请将以下{0}句子翻译成{1}：{2}",
}

TYPE_DICT = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def read_prompt_txt(file_path:str):
    """
    read prompt from text file in line by line manner.
    """
    file_handle = open(file_path)
    prompts = []

    while True:
        line = file_handle.readline()
        if not line:
            break
        line = line.strip()
        prompts.append(line)
    
    return prompts

def read_prompt_json(file_path:str):
    """
    read prompt (dict) from json file in line by line manner.
    """
    file_handle = open(file_path)
    prompts = []
    while True:
        line = file_handle.readline()
        if not line:
            break
        line = line.strip()
        prompts.append(json.loads(line))
    return prompts

FILE_TYPE2LOAD = {
    "txt": read_prompt_txt,
    "json": read_prompt_json
}

def abbreviation2fullname(src_abbreviation, tgt_abbreviation):
    with open("./languages_abbreviation2fullname.txt") as f:
        lines = f.readlines()
    f.close()
    language_fullname_dict = {}
    for line in lines:
        abbreviation = line.strip().split("\t")[0]
        english_full_name = line.strip().split("\t")[1]
        chinese_full_name = line.strip().split("\t")[2]
        language_fullname_dict.update({abbreviation : chinese_full_name}) 
    assert language_fullname_dict[src_abbreviation]!='NONE', f'Source language abbreviation can not convert to Chinese full name, Please check the source language abbreviation in languages_abbreviation2fullname.txt'
    assert language_fullname_dict[tgt_abbreviation]!='NONE', f'Target language abbreviation can not convert to Chinese full name, Please check the target language abbreviation in languages_abbreviation2fullname.txt'
    
    return language_fullname_dict[src_abbreviation], language_fullname_dict[tgt_abbreviation]

def cut2list(line):
    line_cut = jieba.cut(line, cut_all=True)
    line_list = [c for c in line_cut]
    out_list = []
    for c in line_list:
        if len(c) == 0:
            continue
        if c == ' ':
            continue
        out_list.append(c)
    return out_list

def single_prompt(model, tokenizer, prompt="Hello, I'm am conscious and", max_new_tokens:int=128, do_sample:bool=True, num_beams:int=1, top_k:int=50, top_p:float=0.95, no_repeat_ngram_size=6, temperature:float=0.7, cuda=True, verbose=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if cuda:
        # model = model.cuda()
       
        model = tp.tensor_parallel(model)
        input_ids = input_ids.cuda()

    with torch.inference_mode():
        generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams, top_k=top_k, top_p=top_p, temperature=temperature, no_repeat_ngram_size=no_repeat_ngram_size)
    
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False)

    if verbose:
        print(results)
    return results

def batch_prompt(model, tokenizer, prompts:list=["Hello, I'm am conscious and"], max_new_tokens:int=128, do_sample:bool=True, num_beams:int=1, top_k:int=50, top_p:float=0.95, temperature:float=0.7, no_repeat_ngram_size=6, cuda=True, verbose=False):
    tokenizer.padding_side="left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    if cuda:
        # model = model.cuda()

        model = tp.tensor_parallel(model)
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
    with torch.inference_mode():
        generated_ids = model.generate(input_ids,attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=num_beams, top_k=top_k, top_p=top_p, temperature=temperature, no_repeat_ngram_size=no_repeat_ngram_size)
    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
    if verbose:
        print(results)
    return results

def back_process(str_out:str):
    if "\n" in str_out:
        str_out = str_out[:str_out.find("\n")]

    return str_out

def single_translate(model, tokenizer, prompt="Hello, I'm am conscious and", src_lang="zh", tgt_lang="en", with_instruct:bool=True, cuda=True, verbose=False, generation_args:dict=None):
    # src_lang = PROMPT_DICT[src_lang]
    # tgt_lang = PROMPT_DICT[tgt_lang]

    prompt_in = None
    if with_instruct:
        prompt_in = PROMPT_DICT["prompt_input"].format(PROMPT_DICT["translate_instruct"].format(src_lang, tgt_lang, prompt))
    else:
        prompt_in = PROMPT_DICT["translate_prompt"].format(src_lang, tgt_lang, prompt)

    src_len = len(prompt_in)

    if verbose:
        print(f"Translation Prompt: {prompt_in}")

    out_str = single_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_in,
        cuda=cuda,
        verbose=verbose,
        **generation_args,
    )

    translate_res = out_str[0][src_len:]

    if verbose:
        print(f"Translation Result: {translate_res}")

    return translate_res


def fp32to16(model_path,init_model):
    # convert fp32 model to fp16 model (in-place)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, state_dict= torch.load(init_model) if init_model is not None else None)
    torch.save(model.state_dict(), init_model)

# for direct load, in case the state dict needed
def fp32to16_dir(model_path,init_path,tgt_dir):
    # convert fp32 model to fp16 model (in-place)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, state_dict= torch.load(init_path) if init_path is not None else None)
    model.save_pretrained(tgt_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.save_pretrained(tgt_dir)

def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The name of model to use.",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="The path of tokenizer to use.",
    )

    parser.add_argument(
        "--parameter-type",
        type=str,
        # choices=["fp16","bf16","fp32"],
        default="bf16",
        help="The type of model parameters to load.",
    )

    parser.add_argument(
        "--beam-search",
        action='store_true',
        help="Whether to run beam search."
    )

    parser.add_argument(
        "--with-instruct",
        action='store_true',
        help="Whether to run beam search."
    )

    parser.add_argument(
        "--num-beams",
        type=int,
        default=5
    )

    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--out-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--translate",
        action='store_true',
        help="Whether to run translate."
    )

    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Whether to print the details in translation."
    )

    parser.add_argument(
        "--source-language",
        type=str,
        default="zh",
        help="The source language of translation."
    )

    parser.add_argument(
        "--target-language",
        type=str,
        default="en",
        help="The target language of translation."
    )

    parser.add_argument(
        "--translate-json-skip-keys",
        nargs='+',
        default=["answer"],
        help="The key list to skip translation."
    )

    parser.add_argument(
        "--batch-inference",
        action='store_true',
        help="Whether to run inference in batch."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size of run inference in batch."
    )

    parser.add_argument(
        "--split-translate",
        action='store_true',
        help="Whether to run translate by split on dot."
    )


    parser.add_argument(
        "--gold-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--times",
        type=int,
        default=3,
        help="Number of generation for each prompt.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=80,
        help="The configuration top k tokens in the generation of model.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="The configuration top p tokens in the generation of model.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The configuration temperature in the generation of model.",
    )

    args = parser.parse_args()
    set_seed(args.seed)   
    print(args.out_file)

    config = llama.LLaMAConfig.from_pretrained(args.model)
    tokenizer = llama.LLaMATokenizer.from_pretrained(args.tokenizer_path)
    model = llama.LLaMAForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=TYPE_DICT[args.parameter_type],
            config=config,
            state_dict=torch.load(args.checkpoint) if args.checkpoint is not None else None
            )
    
    generation_config = {
        "do_sample": not args.beam_search,
        "num_beams": args.num_beams if args.beam_search else 1,
        "max_new_tokens": args.max_tokens,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature
    }

    # inference for translate
    prompt_file_type = os.path.basename(args.prompt_file).split(".")[-1]

    assert prompt_file_type in FILE_TYPE2LOAD, f"Prompt file type({prompt_file_type}) is not in {FILE_TYPE2LOAD.keys()}"

    prompts = FILE_TYPE2LOAD[prompt_file_type](args.prompt_file)

    # out_handle = open(args.out_file,"w",encoding="utf-8")
    for attr, value in sorted(args.__dict__.items()):
        print(f"\t{attr}={value}")
        # out_handle.write(f"\t{attr}={value}")
    # out_handle.write("\n")

    assert args.source_language != args.target_language, f"Target language({args.target_language}) must be different with the source language({args.source_language})!"

    source_full_name, target_full_name = abbreviation2fullname(args.source_language, args.target_language)
    
    if prompt_file_type == "json":
        for sample in tqdm(prompts):
            tgt_res = {}
            for k in sample.keys():
                if k in args.translate_json_skip_keys or len(sample[k]) == 0:
                    tgt_res[k] = sample[k]
                    continue
                
                tgt_res[k] = single_translate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=sample[k],
                    src_lang=source_full_name,
                    tgt_lang=target_full_name,
                    with_instruct=args.with_instruct,
                    cuda=True,
                    verbose=args.verbose,
                    generation_args=generation_config
                )
            with open(args.out_file, "a", encoding="utf-8") as f:
                print(json.dumps(tgt_res, ensure_ascii=False))
                f.write(json.dumps(tgt_res, ensure_ascii=False))
                f.write("\n")

    else:
        for prompt in tqdm(prompts):            
            tgt_out = single_translate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    src_lang=source_full_name,
                    tgt_lang=target_full_name,
                    with_instruct=args.with_instruct,
                    cuda=True,
                    verbose=args.verbose,
                    generation_args=generation_config
                )

            with open(args.out_file, "a", encoding="utf-8") as f:
                print(tgt_out)
                f.write(tgt_out)
                f.write("\n")
