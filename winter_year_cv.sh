#!/bin/bash
#SBATCH --job-name=machine_learning
#SBATCH --partition=priority
#SBATCH --ntasks=126                     
#SBATCH -x cn[341-342,582]
#SBATCH --nodes=1                            
#BATCH --constraint='epyc128'
#SBATCH --mem-per-cpu=3904    
#SBATCH --mail-type=END  # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=andreas.prevezianos@uconn.edu  # Destination email address
#SBATCH --output=LANL-%x.%j.out # output file formatting %x is the job name and %j the job id

profile=job_${SLURM_JOB_ID}
export profile

python -u /gpfs/homefs1/anp22022/miniconda3/envs/winter_identification/N1_ml_daily_HPC.py
