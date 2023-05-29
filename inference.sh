export CUDA_VISIBLE_DEVICES=0
export SEED=0

export PROMPT_PATH= #Inference_File_Path
export CHECKPOINT_PATH= #Checkpoint_Path
export TOKENIZER_PATH= #Tokenzier_Path

export INSTRUCT=True

# export HIGH_OUT_FILE= #OUT_FILE_PATH
export LOW_OUT_FILE= #OUT_FILE_PATH


export MAX_TOKENS=256
export TOP_K=50
export TOP_P=0.95
export NO_REPEAT_NGRAM_SIZE=6

export HIGH_TEMPERATURE=0.7
export LOW_TEMPERATURE=0.01
export NUM_BEAMS=5

export ADD_PARAMETERS=""
if [ "${INSTRUCT}" != "False" ];
then
ADD_PARAMETERS="--with-instruct "
fi

LOG_FILE="log/llama_13b_inference_local.log"

# HIGH TEPERATURE, MORE CREATIVE

# export OUT_TIME=3
# python -u model/inference.py \
#   --model ${CHECKPOINT_PATH} \
#   --tokenizer-path ${TOKENIZER_PATH} \
#   --prompt-file ${PROMPT_FILE} \
#   ${ADD_PARAMETERS} \
#   --out-file ${HIGH_OUT_FILE} \
#   --seed ${SEED} \
#   --times ${OUT_TIME} \
#   --max-tokens ${MAX_TOKENS} \
#   --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
#   --top-k ${TOP_K} \
#   --top-p ${TOP_P} \
#   --temperature ${HIGH_TEMPERATURE} 2>&1 >>${LOG_FILE}

# LOW TEPERATURE, MORE REALIABLE
export OUT_TIME=1
python -u model/inference.py \
  --model ${CHECKPOINT_PATH} \
  --tokenizer-path ${TOKENIZER_PATH} \
  --prompt-path ${PROMPT_PATH} \
   ${ADD_PARAMETERS} \
  --out-file ${LOW_OUT_FILE} \
  --seed ${SEED} \
  --beam-search \
  --num-beams ${NUM_BEAMS} \
  --times ${OUT_TIME} \
  --max-tokens ${MAX_TOKENS} \
  --no-repeat-ngram-size ${NO_REPEAT_NGRAM_SIZE} \
  --top-k ${TOP_K} \
  --top-p ${TOP_P} \
  --temperature ${LOW_TEMPERATURE} 2>&1 >>${LOG_FILE}