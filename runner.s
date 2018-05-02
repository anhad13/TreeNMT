#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=Runner_NA
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --output=out.STS.%j
python runner.py