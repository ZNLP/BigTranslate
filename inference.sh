export CUDA_VISIBLE_DEVICES=5
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# export CUDA_VISIBLE_DEVICES=5,6,7,8
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9

export SEED=0

export STEP=2800
export MODEL=13b
# DIR_MODEL=ds_seed4120481
# DIR_MODEL=ds_seed4110241
# DIR_MODEL=ins_seed41640
DIR_MODEL=ins_seed21643

# export PROMPT_FILE="/home/wyang/work/MultilingualMT/data/MultilingualMT/top51_test/test_mmt_source_50.txt"
export PROMPT_PATH="/home/wyang/work/MultilingualMT/data/MultilingualMT/top51_test/source/"

export INSTRUCT=True
# export INSTRUCT="False"

export CHECKPOINT_PATH="/home/wyang/work/MultilingualMT/log/13b/${DIR_MODEL}/step${STEP}"
export TOKENIZER_PATH="/home/wyang/work/MultilingualMT/log/13b/${DIR_MODEL}/step${STEP}"

# export HIGH_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b.step${STEP}.txt"
# export HIGH_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b.step${STEP}_ins.txt"
# export HIGH_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b.step${STEP}_cip_long.txt"
# export HIGH_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b.step${STEP}_jttl.txt"
# export HIGH_OUT_FILE="/home/wyang/work/MultilingualMT/log/llama-inference/${DIR_MODEL}/${MODEL}.step${STEP}.txt"
# export LOW_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b_G.step${STEP}.txt"
# export LOW_OUT_FILE="/home/chli/LLaMA/log/llama-inference/13b_B.step${STEP}.txt"
export LOW_OUT_FILE="/home/wyang/work/MultilingualMT/log/llama-inference/${DIR_MODEL}/${STEP}_predict/"
export OUT_TIME=3
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

# if [ ! -d `dirname $HIGH_OUT_FILE` ]; then
#   mkdir `dirname $HIGH_OUT_FILE`
# fi

# if [ ! -d `dirname $LOW_OUT_FILE` ]; then
#   mkdir `dirname $LOW_OUT_FILE`
# fi


LOG_FILE="log/llama_13b_inference_local.log"

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

# beam search is deterministic
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