#!/bin/bash

gpu_id=$1
model=$2
data=$3
tau_s1=$4

select_exp_num=3
two_stage=0

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
                --desc="${model}_s1tau${tau_s1}" --model=${model} \
                --rm_dup_inter=None  \
                --select_exp_num=${select_exp_num} \
                --tau_s1=${tau_s1}  --two_stage=${two_stage}  \
                --split_eval=1  --config_files=${config_files}
else
    python run_recbole.py --gpu_id=${gpu_id} --dataset=${dataset} \
                --desc="${model}_s1tau${tau_s1}" --model=${model} \
                --select_exp_num=${select_exp_num} \
                --tau_s1=${tau_s1}  --two_stage=${two_stage}  \
                --split_eval=1  --config_files=${config_files}
fi

