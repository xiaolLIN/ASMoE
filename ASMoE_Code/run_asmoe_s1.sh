#!/bin/bash

# stage 1: train ASMoE to perform initial learning of input--target patterns

# We take self-attention block to instantiate the experts in ASMoE.

# We provide the command to reproduce the results of stage 1, running ASMoE_SAS_S1
# Amazon Beauty
./asmoe_s1.sh 0 ASMoE_SAS_S1 beauty 0.4
# Amazon_Sports_and_Outdoors
./asmoe_s1.sh 0 ASMoE_SAS_S1 sport 0.6
# Amazon_Toys_and_Games
./asmoe_s1.sh 0 ASMoE_SAS_S1 toy 0.8
# yelp
./asmoe_s1.sh 0 ASMoE_SAS_S1 yelp 0.9


# for hyper-parameter searching of the temperature tau in stage 1
#data=$1
#for tau_s1 in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
#do
#  ./asmoe_s1.sh 0 ASMoE_SAS_S1 ${data} ${tau_s1}
#done