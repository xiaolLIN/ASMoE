#!/bin/bash

gpu_id=$1
model=$2
data=$3
tau_s1=$4
ft_strategy=$5
s1_desc=$6
tau_s2=$7
lora_rank=$8
reg_lmd=$9

select_exp_num=3
lr=0.00005
two_stage=1

if [ $data == "beauty" ] ; then
dataset="Amazon_Beauty"
elif [ $data == "sport" ] ; then
dataset="Amazon_Sports_and_Outdoors"
elif [ $data == "yelp" ] ; then
dataset="yelp"
elif [ $data == "toy" ] ; then
dataset="Amazon_Toys_and_Games"
else
echo "What?"
fi     #ifend


echo ${model}
config_files="configs/${dataset}.yaml"
if [ $data == "yelp" ] ; then
    python run_recbole.py --gpu_id=${gpu_id}  --dataset=${dataset}  \
                --desc="${model}_s1tau${tau_s1}_s2tau${tau_s2}_lora${lora_rank}_reg${reg_lmd}_ft-${ft_strategy}" \
                --tau_s2=${tau_s2} --lora_rank=${lora_rank} --reg_lmd=${reg_lmd} --learning_rate=${lr}  \
                --stage1_model_desc=${s1_desc} \
                --model=${model} --two_stage=${two_stage} --ft_strategy=${ft_strategy} \
                --rm_dup_inter=None  \
                --select_exp_num=${select_exp_num} \
                --tau_s1=${tau_s1}  \
                --split_eval=1 --config_files=${config_files}
else
    python run_recbole.py --gpu_id=${gpu_id} --dataset=${dataset} \
                --desc="${model}_s1tau${tau_s1}_s2tau${tau_s2}_lora${lora_rank}_reg${reg_lmd}_ft-${ft_strategy}" \
                --tau_s2=${tau_s2} --lora_rank=${lora_rank} --reg_lmd=${reg_lmd} --learning_rate=${lr}  \
                --stage1_model_desc=${s1_desc} \
                --model=${model} --two_stage=${two_stage} --ft_strategy=${ft_strategy} \
                --select_exp_num=${select_exp_num} \
                --tau_s1=${tau_s1}  \
                --split_eval=1 --config_files=${config_files}
fi
