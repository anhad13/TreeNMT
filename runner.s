#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=BAL
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --output=out.STS.%j
python runner_dylstm.py --dynet-mem 50000 --dynet-autobatch 1
