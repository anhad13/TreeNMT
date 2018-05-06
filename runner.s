#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=GT.ED
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --output=out.STS.%j
python runner_dylstm.py --dynet-mem 20000 --dynet-autobatch 1
