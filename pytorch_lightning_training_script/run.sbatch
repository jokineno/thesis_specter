#!/bin/bash
#SBATCH --job-name=training_demo
#SBATCH --account=project_2005678
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -o ./puhti_logs/output.%j
#SBATCH -e ./puhti_logs/errors.%j
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user olli.jokinen@helsinki.fi
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:4,nvme:10

#module --force purge 
#module load python-data/3.10

source venv/bin/activate 

echo "Starting to train" 

mkdir -p save 
mkdir -p data 
mkdir -p puhti_logs

echo "===JOB RESULTS===="
srun python train.py \
--save_dir ./save \
--gpus 0 \
--train_file ./data/data-train.p \
--dev_file ./data/data-val.p \
--test_file ./data/data-test.p \
--batch_size 4 \
--num_workers 10 \
--num_epochs 2 \
--grad_accum 256 

echo "===JOB RESULTS===="

echo "=====SLURM_JOBID====="
seff $SLURM_JOBID # this shows some statistics of compute resource utilization
echo "====SLURM_JOBB_ID===="
echo "Finished running train.py" 