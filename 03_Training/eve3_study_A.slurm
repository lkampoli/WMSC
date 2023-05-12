#!/bin/bash --login

echo basename "`pwd`"

myjob=$(basename "`pwd`")
echo $myjob

#SBATCH --job-name=$myjob
#SBATCH --partition=physical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=4GB
#SBATCH --export=NONE

##module purge
##module load openfoam/8
##source $FOAM_BASH

#module unload python/3.7.4
#export PYTHONPATH=""

#module load anaconda3/2021.11
#eval "$(conda shell.bash hook)"
#conda activate eve4bor

module load scipy-bundle/2020.11-python-3.8.2

# Launch multiple process python code
echo "Running"
#python ./my_eve3_model_A.py 2>&1 | tee -a log.my_eve3_model_A.py

