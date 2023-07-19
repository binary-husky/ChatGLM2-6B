PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4


# --preprocessing_num_workers: Specifies the number of worker processes to use for data preprocessing.
# --prompt_column: Indicates the column name in the dataset that contains the input prompt for the model.
# --response_column: Indicates the column name in the dataset that contains the expected response for the given prompt.
# --overwrite_cache: Specifies whether to overwrite the existing dataset cache or not.
# --model_name_or_path: Specifies the pre-trained model name or path to load the model from.
# --output_dir: Specifies the output directory where the generated model and training results will be saved.
# --overwrite_output_dir: Specifies whether to overwrite the existing output directory or not.
# --max_source_length: Specifies the maximum length (in tokens) of the input prompt.
# --max_target_length: Specifies the maximum length (in tokens) of the expected response.
# --per_device_train_batch_size: Specifies the batch size for training on each device (GPU).
# --per_device_eval_batch_size: Specifies the batch size for evaluation on each device (GPU).
# --gradient_accumulation_steps: Specifies the number of gradient accumulation steps to perform before updating the model's parameters.
# --predict_with_generate: Specifies whether to use the model for prediction or not.
# --max_steps: Specifies the maximum number of training steps to perform.
# --logging_steps: Specifies how often (after how many steps) to log training information.
# --save_steps: Specifies how often (after how many steps) to save the model during training.
# --learning_rate: Specifies the learning rate for training the model.
# --pre_seq_len: Specifies the length of the preceding context sequence for the model.
# --quantization_bit: Specifies the number of bits for quantization during training.

#############


PRE_SEQ_LEN=128
LR=1e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/train_gen.json \
    --validation_file AdvertiseGen/train_gen.json \
    --preprocessing_num_workers 20 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/clothgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 20 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

#############


PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
JSON_FILE='t_code.json'

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/$JSON_FILE \
    --validation_file AdvertiseGen/$JSON_FILE \
    --preprocessing_num_workers 20 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/clothgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 20 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

