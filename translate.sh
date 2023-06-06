export CUDA_VISIBLE_DEVICES=0

export SEED=0

#You can find all supported languages abbreviation in languages_abbreviation2fullname.txt
export SRC_LANG= #SOURCE_LANGUAGE_ABBREVIATION (e.g., "en")
export TGT_LANG= #TARGET_LANGUAGE_ABBREVIATION (e.g., "zh")

export PROMPT_FILE= #PROMPT_FILE_PATH (supported file type: txt or json, e.g., "./example/en.txt")

export SAVE_PATH= #SAVE_FILE_PATH


LOG_FILE="translate_bigtrans.example.log"


export INSTRUCT="True"

export VERBOSE="True" #Whether to print the details in translation.

export CHECKPOINT_PATH= #CHECKPOINT_PATH (e.g., /PATH2BigTrans or decapoda-research/llama-7b-hf)
export TOKENIZER_PATH= #TOKENIZER_PATH (e.g., /PATH2BigTrans or decapoda-research/llama-7b-hf)


# export MODEL_TYPE="bf16"
export MODEL_TYPE="fp16" #The type of model parameters to load (e.g., ["fp16", "bf16", "fp32"])

export NUM_BEAMS=5

export MAX_TOKENS=1024
export NO_REPEAT_NGRAM_SIZE=6
export LOW_TEMPERATURE=0.01

export ADD_PARAMETERS=""

if [ "${INSTRUCT}" != "False" ];
then
ADD_PARAMETERS="--with-instruct "
fi

if [ "${VERBOSE}" != "False" ];
then
ADD_PARAMETERS="${ADD_PARAMETERS} --verbose "
fi


# beam search is deterministic
export OUT_TIME=1
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
