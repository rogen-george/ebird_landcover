#!/bin/bash
#SBATCH -J land_cover
#SBATCH -A eecs
#SBATCH -o run.out
#SBATCH -e run.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=georgrog@oregonstate.edu
#SBATCH -t 2-12:30:00

cd /nfs/hpc/share/georgrog/ebird/ebird_landcover
source py3/bin/activate
cd /nfs/hpc/share/georgrog/research/landcover
python driver.py
deactivate