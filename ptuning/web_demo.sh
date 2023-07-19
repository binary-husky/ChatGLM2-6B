PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=4 python3 web_demo.py \
    --model_name_or_path THUDM/chatglm2-6b \
    --ptuning_checkpoint output/clothgen-chatglm2-6b-pt-128-1e-2/checkpoint-100 \
    --pre_seq_len $PRE_SEQ_LEN

