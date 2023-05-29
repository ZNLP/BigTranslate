from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from nltk.translate.bleu_score import sentence_bleu
import jieba
import llama
import argparse
from accelerate.utils import set_seed

import os


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
}


def read_prompt(file_path:str):
    file_handle = open(file_path)
    prompts = []

    while True:
        line = file_handle.readline()
        if not line:
            break
        line = line.strip()
        prompts.append(line)
    
    return prompts


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
        model = model.cuda()
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
        model = model.cuda()
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

def eval_translate(model, tokenizer, prompts, gold_path, context="Chinese: 我想回家。\nEnglish: I want to go home.\n\nChinese: 我不知道。\nEnglish: I don't know.\n\nChinese: {}\nEnglish: ", cuda=True, split_translate=False, generation_args:dict=None):
    file_handler = open(gold_path,"r",encoding="utf-8")

    total_b = 0
    for prompt in prompts:
        
        s_t = time.time()
        if not split_translate:
            str_in = context.format(prompt)
            print(f"Input:{str_in}\n")
            decode_res = single_prompt(model=model, tokenizer=tokenizer, prompt=str_in, cuda=cuda, verbose=False, **generation_args)
            predict_res = back_process(decode_res[0][len(str_in):])
        else:
            inputs = prompt.split("，")
            preds = []
            for str_id, str_in in enumerate(inputs):
                if len(str_in) == 0:
                    continue
                str_in = context.format(str_in)
                print(f"Input-{str_id+1}:{str_in}\n")
                
                decode_res = single_prompt(model=model, tokenizer=tokenizer, prompt=str_in, cuda=cuda, verbose=False, **generation_args)
                decode_res = back_process(decode_res[0][len(str_in):])
                
                print(f"Output-{str_id+1}:{decode_res}\n")
                preds.append(decode_res)
            
            predict_res = ", ".join(preds)

        gold_line = file_handler.readline()

        print("Output:",predict_res)
        print("Gold:",gold_line)
        
        gold_list = cut2list(gold_line)
        
        predict_list = cut2list(predict_res)
        
        curr_b = sentence_bleu([gold_list], predict_list)
        total_b += curr_b

        e_t = time.time()
        print(f"Time cost:{e_t-s_t}s, bleu:{curr_b} \n\n")

    file_handler.close()

    print(f"Average bleu: {total_b/len(prompts)}")

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
        "--prompt-path",
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
            torch_dtype=torch.float16,
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


    if args.prompt_path is not None:
        for file in os.listdir(args.prompt_path):
            print(f"Inference {file}...")
            prompts = read_prompt(os.path.join(args.prompt_path,file))
            print(len(prompts))
            if args.translate:
                # do translation
                eval_translate(model=model, tokenizer=tokenizer, prompts=prompts, gold_path=args.gold_file, cuda=True, split_translate=args.split_translate, generation_args=generation_config)
            elif args.batch_inference:
                # batch inference
                out_handle = open(args.out_file,"w",encoding="utf-8")
                for attr, value in sorted(args.__dict__.items()):
                    print(f"\t{attr}={value}")
                    # out_handle.write(f"\t{attr}={value}")
                t_id = 1
                prompt_list = to_matrix(prompts, args.batch_size)
                for prompts_in in prompt_list:
                    outputs = batch_prompt(model, tokenizer, prompts=prompts_in, cuda=True, **generation_config)
                    for out_str in outputs:
                        out_handle.write(f"T-{t_id} {out_str}\n")
                        t_id += 1
                out_handle.close()
            else:
                # inference
                out_handle = open(os.path.join(args.out_file, file),"w",encoding="utf-8")
                for attr, value in sorted(args.__dict__.items()):
                    print(f"\t{attr}={value}")
                    # out_handle.write(f"\t{attr}={value}")

                for prompt in prompts:            
                    if args.with_instruct:
                        prompt = PROMPT_DICT["prompt_input"].format(prompt)

                    # out_handle.write(f"\n*****Input: {prompt}\n")

                    for i in range(args.times):
                        s_t = time.time()
                        results = single_prompt(model=model, tokenizer=tokenizer, prompt=prompt, cuda=True, **generation_config)
                        step_time = time.time() - s_t

                        out_handle.write(f"\n*****Output(Time-{i+1},cost {step_time:.2f}s): ")
                        out_handle.write(results[0]+"\n")

                out_handle.close()
    else:
        prompts = read_prompt(args.prompt_file)

        if args.translate:
            # do translation
            eval_translate(model=model, tokenizer=tokenizer, prompts=prompts, gold_path=args.gold_file, cuda=True, split_translate=args.split_translate, generation_args=generation_config)
        elif args.batch_inference:
            # batch inference
            out_handle = open(args.out_file,"w",encoding="utf-8")
            for attr, value in sorted(args.__dict__.items()):
                print(f"\t{attr}={value}")
                # out_handle.write(f"\t{attr}={value}")
            t_id = 1
            prompt_list = to_matrix(prompts, args.batch_size)
            for prompts_in in prompt_list:
                outputs = batch_prompt(model, tokenizer, prompts=prompts_in, cuda=True, **generation_config)
                for out_str in outputs:
                    out_handle.write(f"T-{t_id} {out_str}\n")
                    t_id += 1

            out_handle.close()
        else:
            # inference
            out_handle = open(args.out_file,"w",encoding="utf-8")
            for attr, value in sorted(args.__dict__.items()):
                print(f"\t{attr}={value}")
                out_handle.write(f"\t{attr}={value}")

            for prompt in prompts:            
                if args.with_instruct:
                    prompt = PROMPT_DICT["prompt_input"].format(prompt)

                # out_handle.write(f"\n*****Input: {prompt}\n")

                for i in range(args.times):
                    s_t = time.time()
                    results = single_prompt(model=model, tokenizer=tokenizer, prompt=prompt, cuda=True, **generation_config)
                    step_time = time.time() - s_t

                    out_handle.write(f"\n*****Output(Time-{i+1},cost {step_time:.2f}s): ")
                    out_handle.write(results[0]+"\n")

            out_handle.close()