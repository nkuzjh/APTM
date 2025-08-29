#!/bin/bash

# 设置基础参数
# num_gpus=1
log_dir="./logs"
mkdir -p $log_dir

dataset_name="itr_rstp"

# 定义实验名
exp_names=(
    "tta_exp6"
    "tta_exp6.0.1"
    "tta_exp6.0.2"
    "tta_exp6.1"
    "tta_exp6.1.1"
    "tta_exp6.1.2"
    "tta_exp6.2"
    "tta_exp6.2.1"
    "tta_exp6.2.2"
    "tta_exp6.3"
    "tta_exp6.3.1"
    "tta_exp6.3.2"
    "tta_exp6.4"
    "tta_exp6.4.1"
    "tta_exp6.4.2"
    "tta_exp6.5"
    "tta_exp6.5.1"
    "tta_exp6.5.2"
    "tta_exp6.6"
    "tta_exp6.6.1"
    "tta_exp6.6.2"
)

# 按顺序执行每个训练任务
for i in "${!exp_names[@]}"
do
    exp_name=${exp_names[$i]}

    echo " "
    echo "Starting Training: $exp_name ..."
    start_time=$(date +%s)
    echo "Start Time: $(date +"%Y-%m-%d %T")"


    # python3 run_tta.py --device_no "0" --task "itr_rstp" --tta --config tta_configs/TTA_Retrieval_rstp/exp_debug.yaml --output_dir "output/ft_rstp/tta/exp_debug" --checkpoint "checkpoints/ft_rstp/checkpoint_best.pth"
    nohup python3 run_tta.py --device_no "0" --task $dataset_name --tta  --config "tta_configs/TTA_Retrieval_rstp/$exp_name.yaml" --output_dir "output/ft_rstp/tta/$exp_name" --checkpoint "checkpoints/ft_rstp/checkpoint_best.pth" > "logs/TTA_Retrieval_rstp/$exp_name.log" 2>&1 &


    # 等待当前任务完成
    wait

    # 记录结束时间
    end_time=$(date +%s)
    duration=$(( end_time - start_time ))
    # 格式化时间（分钟和秒）
    minutes=$(( duration / 60 ))
    seconds=$(( duration % 60 ))
    # 结束时间和运行时长
    echo "End Time: $(date +"%Y-%m-%d %T")"
    echo "Duration: ${minutes}m${seconds}s"
    echo "Finsh Training $exp_name ."
done
