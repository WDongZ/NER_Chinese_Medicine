#!/bin/bash
#SBATCH --job-name=peft_train            # 作业名字
#SBATCH --partition=gpu                  # 分区（GPU节点）
#SBATCH --gres=gpu:a100-80:1                     # 指定GPU型号和数量
#SBATCH --mem=64G                        # 内存大小
#SBATCH --time=04:00:00                  # 最长运行时间（小时:分钟:秒）
#SBATCH --output=logs/%x_%j.out          # 输出日志文件

# 加载环境变量（必要）
source ~/.bashrc
conda activate peft                     # 启用你的conda环境

# 跳转到工作目录
cd /home/e/e1518642/medical

# 运行Python脚本
python train.py
