#!/bin/bash
#$ -M mfigura@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -N DAC,lr=0.1,0.01,rs=100
#$ -o info
module load conda
conda activate MARL_env
/afs/crc.nd.edu/user/m/mfigura/.conda/envs/MARL_env/bin/python /afs/crc.nd.edu/user/m/mfigura/Private/Cooperative_MARL/Decentralized_AC/main.py --max_ep_len=100 --actor_lr=0.1 --critic_lr=0.01 --update_frequency=1 --communication_frequency=1 --graph_diam=4 --summary_dir='/afs/crc.nd.edu/user/m/mfigura/Private/Cooperative_MARL/Decentralized_AC/my_scripts/test_name=DAC/max_ep_len=100/upd_freq=1,graph_diam=4,lr=0.1,0.01/seed=100' --random_seed=100 > out.txt
