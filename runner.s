#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=GT.ED
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --output=out.STS.%j
python runner_dylstm.py --dynet-mem 80000 #--dynet-autobatch 1
