#!/bin/bash

# stage 2: continue to train ASMoE based on the trained model in the last stage.

# We take self-attention block to instantiate the experts in ASMoE.

# We provide the command to reproduce the results of stage 2, running ASMoE_SAS_S2
# Amazon Beauty
./asmoe_s2.sh 0 ASMoE_SAS_S2 beauty 0.4 emb_lora_enc ASMoE_SAS_S1_s1tau0.4 1 8 0.009
# Amazon_Sports_and_Outdoors
./asmoe_s2.sh 0 ASMoE_SAS_S2 sport 0.6 emb_lora_enc ASMoE_SAS_S1_s1tau0.6 1 16 0.001
# Amazon_Toys_and_Games
./asmoe_s2.sh 0 ASMoE_SAS_S2 toy 0.8 emb_lora_enc ASMoE_SAS_S1_s1tau0.8 1 32 0.009
# yelp
./asmoe_s2.sh 0 ASMoE_SAS_S2 yelp 0.9 emb_lora_enc ASMoE_SAS_S1_s1tau0.9 0.1 16 0.003


# hyper-parameter searching in stage 2: tau_s2, lora_rank, reg_lmd
#data=$1
#
#if [ $data == "beauty" ] ; then
#s1_desc='ASMoE_SAS_S1_s1tau0.4'
#tau_s1=0.4
#elif [ $data == "sport" ] ; then
#s1_desc='ASMoE_SAS_S1_s1tau0.6'
#tau_s1=0.6
#elif [ $data == "toy" ] ; then
#s1_desc='ASMoE_SAS_S1_s1tau0.8'
#tau_s1=0.8
#elif [ $data == "yelp" ] ; then
#s1_desc='ASMoE_SAS_S1_s1tau0.9'
#tau_s1=0.9
#else
#echo "What?"
#fi     #ifend
#
#for tau_s2 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
#do
#  for lora_rank in 8 16 32;
#  do
#    for reg_lmd in 0.001 0.005 0.01 0.05 0.1 0.5;         # need a further finer search
#    do
#      ./asmoe_s2.sh 0 ASMoE_SAS_S2 ${data} ${tau_s1} emb_lora_enc ${s1_desc} ${tau_s2} ${lora_rank} ${reg_lmd}
#    done
#  done
#done