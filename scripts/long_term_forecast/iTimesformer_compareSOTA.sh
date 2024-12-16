export CUDA_VISIBLE_DEVICES=0


#General settings

# Check if pred_len argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pred_len>"
    echo "Example: $0 96"
    exit 1
fi

pred_len=$1

echo "Using prediction length: $pred_len"

model_names=(
    "iTimesformerCyclicAttn"
    #"iTimesformer"
    "iTransformer"
    "PatchTST"
    "DLinear"
    #"TimesNet"
    "TimeMixer"
    "TimeXer"
)


#Datasets:

# Electricity
#_______________________________________________________
seq_lens=(24 48 96 192)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id "Electricity_${pred_len}" \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 50 \
        --dec_in 50 \
        --c_out 50 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
done
#_______________________________________________________

# Weather
#_______________________________________________________
seq_lens=(144 288 576 1152)
#seq_lens=(1152)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id "Weather_${pred_len}" \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 21 \
        --dec_in 21 \
        --c_out 21 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 144 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
done
#_______________________________________________________


# Traffic
#_______________________________________________________
seq_lens=(24 48 96 192)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id "Traffic_${pred_len}" \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 50 \
        --dec_in 50 \
        --c_out 50 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
done
#_______________________________________________________

# ETTh1
#_______________________________________________________
seq_lens=(24 48 96 192)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTh2.csv \
        --model_id "ETTh2_${pred_len}" \
        --model $model \
        --data ETTh2 \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
done

#_______________________________________________________

# ETTm2
#_______________________________________________________
seq_lens=(96 192 384 768)
#seq_lens=(1152)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id "ETTm2_${pred_len}" \
        --model $model \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 96 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2
    done
done







