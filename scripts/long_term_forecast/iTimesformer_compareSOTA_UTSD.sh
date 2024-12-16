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
    #"iTransformer"
    #"PatchTST"
    #"DLinear"
    #"TimesNet"
    #"TimeMixer"
    #"TimeXer"
)



#Datasets:

# KDDCup2018
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Nature/kdd_cup_2018_dataset_without_missing_values \
        --data_path random.csv \
        --model_id "KDDCup2018_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# ERA5Surface
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Nature/ERA5/surface \
        --data_path random.csv \
        --model_id "ERA5Surface_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 75 \
        --dec_in 75 \
        --c_out 75 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 8 \
        --split 0.8 \
        --stride 10 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# AustrailianElectricityDemand
#_______________________________________________________
seq_lens=(48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Energy/australian_electricity_demand_dataset \
        --data_path random.csv \
        --model_id "AustrailianElectricityDemand_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 48 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# BenzeneConcentration
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Environment/BenzeneConcentration \
        --data_path random.csv \
        --model_id "BenzeneConcentration_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 8 \
        --dec_in 8 \
        --c_out 8 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# MotorImagery
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Health/MotorImagery \
        --data_path random.csv \
        --model_id "MotorImagery_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 64 \
        --dec_in 64 \
        --c_out 64 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# TDBrain
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Health/TDBrain_csv \
        --data_path random.csv \
        --model_id "TDBrain_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 33 \
        --dec_in 33 \
        --c_out 33 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# LondonSmartMeters
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Energy/london_smart_meters_dataset_without_missing_values \
        --data_path random.csv \
        --model_id "LondonSmartMeters_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# AustraliaRainfall
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Environment/AustraliaRainfall \
        --data_path random.csv \
        --model_id "AustraliaRainfall_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 3 \
        --dec_in 3 \
        --c_out 3 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# BeijingAir
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Environment/BeijingPM25Quality \
        --data_path random.csv \
        --model_id "BeijingAir_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 9 \
        --dec_in 9 \
        --c_out 9 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________

# PedestrianCounts
#_______________________________________________________
seq_lens=(24 48 96 192 672)

# Loop through array
for model in "${model_names[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/UTSD-full-npy/Transport/pedestrian_counts_dataset \
        --data_path random.csv \
        --model_id "PedestrianCounts_${pred_len}" \
        --model $model \
        --data UTSD \
        --features M \
        --seq_len $seq_len \
        --label_len 24 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --main_cycle 24 \
        --split 0.8 \
        --stride 100 \
        --down_sampling_layers 3 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --full_mlp True \
        --model_trend True
    done
done
#_______________________________________________________







