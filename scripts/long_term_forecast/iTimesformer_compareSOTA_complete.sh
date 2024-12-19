#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# General settings

# Check if pred_len and script_name arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <pred_len> <script_name>"
    echo "Example: $0 96 run.py"
    exit 1
fi

pred_len=$1
script_name=$2

echo "Using prediction length: $pred_len"
echo "Using script: $script_name"

model_names=(
    "DLinear"
    "iTimesformerCyclicAttn"
    "iTransformer"
    "PatchTST"
    "TimeMixer"
    "TimeXer"
)

# Datasets and their specific settings
declare -A datasets=(
    #Datasets that will be used in the final experiments
    #["ERA5Surface"]="input_dims=75 main_cycle=8 root_path=./dataset/UTSD-full-npy/Nature/ERA5/surface stride=10 data_path=random.csv data=UTSD"
    #["ERA5Pressure"]="input_dims=75 main_cycle=8 root_path=./dataset/UTSD-full-npy/Nature/ERA5/pressure stride=10 data_path=random.csv data=UTSD"
    ["BenzeneConcentration"]="input_dims=8 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/BenzeneConcentration stride=100 data_path=random.csv data=UTSD"
    ["MotorImagery"]="input_dims=64 main_cycle=96 root_path=./dataset/UTSD-full-npy/Health/MotorImagery stride=100 data_path=random.csv data=UTSD"
    ["TDBrain"]="input_dims=33 main_cycle=48 root_path=./dataset/UTSD-full-npy/Health/TDBrain_csv stride=100 data_path=random.csv data=UTSD"
    ["BeijingAir"]="input_dims=9 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/BeijingPM25Quality stride=100 data_path=random.csv data=UTSD"
    #["Electricity"]="input_dims=50 main_cycle=24 root_path=./dataset/electricity/ stride=1 data_path=electricity.csv data=custom"
    ["Weather"]="input_dims=21 main_cycle=24 root_path=./dataset/weather/ stride=1 data_path=weather.csv data=custom"
    #["Traffic"]="input_dims=50 main_cycle=24 root_path=./dataset/traffic/ stride=1 data_path=traffic.csv data=custom"
    ["ETTh1"]="input_dims=7 main_cycle=24 root_path=./dataset/ETT-small/ stride=1 data_path=ETTh1.csv data=ETTh1"
    ["ETTm1"]="input_dims=7 main_cycle=96 root_path=./dataset/ETT-small/ stride=1 data_path=ETTm1.csv data=custom"
    ["ETTh2"]="input_dims=7 main_cycle=24 root_path=./dataset/ETT-small/ stride=1 data_path=ETTh2.csv data=ETTh1"
    ["ETTm2"]="input_dims=7 main_cycle=96 root_path=./dataset/ETT-small/ stride=1 data_path=ETTm2.csv data=custom"
    ["Exchange"]="input_dims=8 main_cycle=96 root_path=./dataset/exchange_rate/ stride=1 data_path=exchange_rate.csv data=custom"
    #Other datasets
    #["AustrailianElectricityDemand"]="input_dims=1 main_cycle=48 root_path=./dataset/UTSD-full-npy/Energy/australian_electricity_demand_dataset stride=100 data_path=random.csv data=UTSD"
    #["AustraliaRainfall"]="input_dims=3 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/AustraliaRainfall stride=100 data_path=random.csv data=UTSD"
    #["KDDCup2018"]="input_dims=1 main_cycle=24 root_path=./dataset/UTSD-full-npy/Nature/kdd_cup_2018_dataset_without_missing_values stride=100 data_path=random.csv data=UTSD"
    #["PedestrianCounts"]="input_dims=1 main_cycle=24 root_path=./dataset/UTSD-full-npy/Transport/pedestrian_counts_dataset stride=100 data_path=random.csv data=UTSD"
    #["LondonSmartMeters"]="input_dims=1 main_cycle=24 root_path=./dataset/UTSD-full-npy/Energy/london_smart_meters_dataset_without_missing_values stride=1000 data_path=random.csv data=UTSD"
)

# Sequence length options (same for all datasets)
seq_lens=(720)

# Common settings
label_len=24
features="M"
e_layers=2
d_layers=1
factor=3
d_model=128
d_ff=128
itr=1
split=0.8
down_sampling_layers=3
down_sampling_method="avg"
down_sampling_window=2
des="Exp"
results_file="results_hpo.txt"

# Loop through datasets
for dataset in "${!datasets[@]}"; do
    # Parse dataset-specific settings
    eval ${datasets[$dataset]}
    for seq_len in "${seq_lens[@]}"; do
        for model in "${model_names[@]}"; do
            python -u $script_name \
                --model_id "${dataset}_${pred_len}" \
                --results_file $results_file \
                --is_training 1 \
                --task_name long_term_forecast \
                --root_path $root_path \
                --data_path $data_path \
                --model $model \
                --data $data \
                --features $features \
                --seq_len $seq_len \
                --label_len $label_len \
                --pred_len $pred_len \
                --e_layers $e_layers \
                --d_layers $d_layers \
                --factor $factor \
                --enc_in $input_dims \
                --dec_in $input_dims \
                --c_out $input_dims \
                --des $des \
                --d_model $d_model \
                --d_ff $d_ff \
                --itr $itr \
                --main_cycle $main_cycle \
                --patch_len $main_cycle \
                --split $split \
                --stride $stride \
                --pstride $main_cycle \
                --down_sampling_layers $down_sampling_layers \
                --down_sampling_method $down_sampling_method \
                --down_sampling_window $down_sampling_window \
                --full_mlp
        done
    done
done