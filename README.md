# 🦙  **BigTranslate**  🚀
## 📢 News
**💥[June 6, 2023] We have updated [translate.sh](./translate.sh) to directly utilize BigTranslate for multilingual machine translation! Just only need to provide source and target language abbreviation, file path in script.**

[May 31, 2023] We have released BigTranslate model on [Hugging Face](https://huggingface.co/James-WYang/BigTrans) for download and experience!
## 📝 Case Study
*French -> English*
> Instruction: 请将以下法语句子翻译成英语：Cette découverte permet également de mieux comprendre l'évolution des plumes chez les oiseaux.
> 
> BigTranslate: This finding also provides a better understanding of the evolution of feathers in birds.
> 
> Instruction: 请将以下句子翻译成英语：Il a été volé par des pirates, attaqué au Tibet par un chien enragé, a échappé à un mariage au Népal et a été arrêté en Inde.
> 
> BigTranslate: Along the way he was hijacked by pirates, attacked by an angry dog in Tibet, narrowly escaped an arranged marriage in Nepal, and was arrested in India.

*Danish -> English*
> Instruction: 请将以下丹麦语句子翻译成英语：Rapporten starter med at opfordre til en åben debat, og til at der skabes enighed i USA om landets politik over for Mellemøsten.
> 
> BigTranslate: The report starts by calling for an open debate and for unity in the United States on its policy towards the Middle East.
> 
> Instruction: 请将以下丹麦语句子翻译成英语：Mere end fire millioner mennesker tog til Rom for at deltage i begravelsen
> 
> BigTranslate: More than 4 million people went to Rome to attend the funeral.

*English -> Chinese*
> Instruction: 请将以下英语句子翻译成汉语：You may also wish to consult the advice of governments other than your own, but their advice is designed for their citizens.
> 
> BigTranslate: 您也可以咨询其他国家政府的意见，但这些意见是针对本国公民而设计的。
> 
> Instruction: 请将以下句子翻译成汉语：Typically there will be a tuition fee to enroll in these educational programs.
> 
> BigTranslate: 参加这些教育培训项目的学生通常需要缴纳学费。

*Tibetan -> Chinese*
> Instruction: 请将以下句子翻译成汉语：ཚད་ལྡན་གྱི་འཁྱགས་ཤུད་སྤོ་ལོའི་ཐང་གི་ཚད་གཞི་ཆེ་ཤོས་ནི་རིང་ཚད་ལ་སྨི་61དང་ཞེང་ཚད་ལ་སྨི་30ཡོད།
> 
> BigTranslate: 标准冰橇长度最大的是61米，最小的是30米
> 
> Instruction: 请将以下藏语句子翻译成汉语：ངས་ཤེས་གསལ་ལྟར་ན།  ང་ཚོའི་རྐང་རྩེད་སྤོ་ལོ་རུ་ཁག་གི་ནུས་ཤུགས་ཁོ་ཚོ་ལས་བཟང་།
> 
> BigTranslate: 就我所知，我们的足球队比他们强。

*English -> Portuguese*
> Instruction: 请将以下英语句子翻译成葡萄牙语：Several large television screens were installed in various places in Rome to let the people watch the ceremony.
> 
> BigTranslate: Diversos grandes ecrãs televisivos foram instalados em diversos lugares em Roma para que as pessoas pudessem assistir à cerimónia.
> 
> Instruction: 请将以下英语句子翻译成葡萄牙语：Scientists say the explosion caused by the collision was massive.
> 
> BigTranslate: Os cientistas dizem que a explosão causada pela colisão foi massiva.

*English -> Swedish*
> Instruction: 请将以下句子翻译成瑞典语：Negotiators tried to rectify the situation, but the prisoners' demands are not clear.
> 
> BigTranslate: Förhandlarna försöker korrigera situationen, men fångarnas krav är inte klara.
> 
> Instruction: 请将以下英语句子翻译成瑞典语：Although the water level will only rise a few feet after the flood, officials are hoping it will be enough to restore eroded sandbars downstream.
> 
> BigTranslate: Även om vattennivån endast ökar några fot efter översvämningen, hoppas myndigheterna att det räcker för att återställa eroderade sandbankar nedströms.

## ⭐ BigTranslate Construction
### 🌓 Large-scale Parallel Dataset Construction
In order to enhance the language capabilities of the Chinese LLaMA model to support 102 languages, we constructed a comprehensive parallel corpus dataset consisting of 102 languages. This dataset was employed to continue training the foundational model. The compilation of this dataset drew upon multiple sources, including widely available public parallel corpus datasets and household datasets. The public datasets utilized in our study contain IWSLT, WMT, CCMT, and OPUS-100, forming the initial corpus of our dataset.

To effectively illustrate the distribution of the corpus, we present a visual representation of the language-pair distribution within the multilingual datasets. The matter pertaining to the imbalance between high-resource and low-resource language pairs continues to be a prominent concern within the current corpus.

![image](./pics/corpus_distribution.png)

### 🌔 Incremental Multilingual Pre-training
In this incremental pre-training method, we gradually expose the model to language pairs in a curriculum-like manner. Initially, the model is exposed to high-resource language pairs, allowing it to establish a solid foundation in those languages. Subsequently, we progressively introduce low-resource language pairs, enabling the model to gradually expand its knowledge and proficiency in these languages.

Specifically, we follow a three-step approach in our incremental pre-training method. Firstly, we set the sample interval size and divide language pairs into distinct intervals based on the number of instances for each language pair. Secondly, we calculate the sample mean for all language pairs in each interval. Thirdly, we dynamically measure the moment of adding the language-pair samples next interval according to the sample mean in the previous sample interval. In the following part, we detail the three steps.

![image](./pics/The_outline_of_Increment_pre-training.png)

### 🌕 Multilingual Translation Instruction Tuning

We have designed a set of 28 multilingual translation prompts that encompass various application scenarios for multilingual translation. We randomly select a prompt from the set for instruction tuning for each parallel sentence. Accordingly, the instruction tuning dataset is scrambled to ensure randomness and diversity.

During training phase, We randomly select a prompt from the following 28 multilingual translation prompts for each sentence.
```
请将以下{SRC_LANG}句子翻译成{TGT_LANG}：{SRC_Text}
请将以下{SRC_LANG}文本翻译成{TGT_LANG}：{SRC_Text}
请将以下句子翻译成{TGT_LANG}：{SRC_Text}
请将以下文本翻译成{TGT_LANG}：{SRC_Text}
请提供{SRC_LANG}句子“{SRC_Text}”的{TGT_LANG}翻译
请提供{SRC_LANG}文本“{SRC_Text}”的{TGT_LANG}翻译
请提供句子“{SRC_Text}”的{TGT_LANG}翻译
请提供文本“{SRC_Text}”的{TGT_LANG}翻译
以下{SRC_LANG}句子“{SRC_Text}”用{TGT_LANG}如何表达
以下{SRC_LANG}文本“{SRC_Text}”用{TGT_LANG}如何表达
以下句子“{SRC_Text}”用{TGT_LANG}如何表达
以下文本“{SRC_Text}”用{TGT_LANG}如何表达
以下{SRC_LANG}句子“{SRC_Text}”的{TGT_LANG}翻译是什么？
以下{SRC_LANG}文本“{SRC_Text}”的{TGT_LANG}翻译是什么？
以下句子“{SRC_Text}”的{TGT_LANG}翻译是什么？
以下文本“{SRC_Text}”的{TGT_LANG}翻译是什么？
请生成以下{SRC_LANG}句子“{SRC_Text}”的{TGT_LANG}翻译
请生成以下{SRC_LANG}文本“{SRC_Text}”的{TGT_LANG}翻译
请生成以下句子“{SRC_Text}”的{TGT_LANG}翻译
请生成以下文本“{SRC_Text}”的{TGT_LANG}翻译
如何用{TGT_LANG}表达{SRC_LANG}句子“{SRC_Text}”
如何用{TGT_LANG}表达{SRC_LANG}文本“{SRC_Text}”
如何用{TGT_LANG}表达句子“{SRC_Text}”
如何用{TGT_LANG}表达文本“{SRC_Text}”
这个{SRC_LANG}句子“{SRC_Text}”用{TGT_LANG}怎么说？
这个{SRC_LANG}文本“{SRC_Text}”用{TGT_LANG}怎么说？
这个句子“{SRC_Text}”用{TGT_LANG}怎么说？
这个文本“{SRC_Text}”用{TGT_LANG}怎么说？
```
During inference phase, We randomly select a prompt from the following two multilingual translation prompts for each sentence.
```
请将以下{SRC_LANG}句子翻译成{TGT_LANG}：{SRC_Text}
请将以下句子翻译成{TGT_LANG}：{SRC_Text}
```


## 🌟 Experiments
### 🌖 Automatic Evaluation with BLEU
An illustrated comparison of 102 languages from X to English or Chinese between BigTranslate, ChatGPT and Google Translate. We sort the language scores in BLEU for BigTranslate in descending order.

![image](./pics/104langs_bleu.png)

### 🌗 Automatic Evaluation with GPT-4
An illustrated comparison of 70 languages from X to English or Chinese between BigTranslate, ChatGPT and Google Translate. We sort the language scores in GPT-4 score for BigTranslate in descending order.

![image](./pics/70langs_gpt4.png)

##  🤖 BigTranslate Model

### ⚠️ User Notice (Must Read)

<!-- The official [LLaMA models released by Facebook prohibit commercial use](https://github.com/facebookresearch/llama), and the official model weights have not been open-sourced (although there are many third-party download links available online). -->

The BigTranslate Model weights are based on [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) protocols, which is only for research use and cannot be used for commercial purposes. 

***Please confirm that you are using the model in this warehouse with [permission](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form).***

### 📎 Model Download

**BigTranslate**：[Hugging Face](https://huggingface.co/James-WYang/BigTrans) 
<!-- [Google Drive](https://drive.google.com/drive/folders/1r_X7sehOZ1g_an26EziuOrf7G8Q0DjB_?usp=drive_link) -->

<!-- > ⏳ Model is uploading -->

### 📌 Model Inference
Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Example usage:

  ```bash
  python -u model/inference.py \
    --model ${CHECKPOINT_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --prompt-file ${PROMPT_FILE} \
    --with-instruct \
    --out-file ${LOW_OUT_FILE} \
    --seed ${SEED} \
    --beam-search \
    --num-beams ${NUM_BEAMS} \
    --times ${OUT_TIME} \
    --max-tokens ${MAX_TOKENS} \
    --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
    --top-k ${TOP_K} \
    --top-p ${TOP_P} \
    --temperature ${TEMPERATURE} 2>&1 >>${LOG_FILE}
  ```
We can customize the hyperparameters:

  ```bash
  python -u model/inference.py \
    --model ${CHECKPOINT_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --prompt-file ${PROMPT_FILE} \
    --with-instruct \
    --out-file ${BEAM_OUT_FILE} \
    --seed ${SEED} \
    --beam-search \
    --num-beams 5 \
    --times 1 \
    --max-tokens 256 \
    --no-repeat-ngram-size 6 2>&1 >>${LOG_FILE}
  ```
We made a script in [inference.sh](./inference.sh) to run model inference.

### 💡 Translate with BigTranslate

Example usage:
  ```
  python -u model/translate.py \
    --model ${CHECKPOINT_PATH} \
    --tokenizer-path ${TOKENIZER_PATH} \
    --prompt-file ${PROMPT_FILE} \
    ${ADD_PARAMETERS} \
    --out-file ${SAVE_PATH} \
    --source-language ${SRC_LANG} \
    --target-language ${TGT_LANG} \
    --seed ${SEED} \
    --beam-search \
    --parameter-type ${MODEL_TYPE} \
    --num-beams ${NUM_BEAMS} \
    --times ${OUT_TIME} \
    --max-tokens ${MAX_TOKENS} \
    --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
    --temperature ${LOW_TEMPERATURE} 2>&1 >>${LOG_FILE}
  ```
We made a script in [translate.sh](./translate.sh) to translate with BigTranslate.

## License

Our code and documents are released under Apache Licence 2.0

Following LLaMA, our pre-trained weights are released under GNU General Public License v3.0

## Acknowledgement

We thank all contributors for BigTranslate projects.

This repo benefits from [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). Thanks for their wonderful works.

## Contact

If you have any questions, please feel free to contact us by sending an email to {yangwen2023, lichong2021}@ia.ac.cn, {jjzhang, cqzong}@nlpr.ia.ac.cn .

## Citation

```
@article{yang-etal-2023-BigTrans,
  author    = {Wen Yang and
               Chong Li and
               Jiajun Zhang and
               Chengqing Zong},
  title={BigTrans: Augmenting Large Language Models with Multilingual Translation Capability over 100 Languages},
  journal={arXiv preprint arXiv:2305.18098},
  url={https://arxiv.org/abs/2305.18098},
  year={2023}
}
```




