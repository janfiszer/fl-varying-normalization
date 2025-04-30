#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J TDtsTest
# Liczba alokowanych wÄ™zĹ‚Ăłw
#SBATCH -n 1
## Liczba zadaĹ„ per wÄ™zeĹ‚ (domyĹ›lnie jest to liczba alokowanych rdzeni na wÄ™Ĺşle)
#SBATCH --ntasks-per-node=1
## IloĹ›Ä‡ pamiÄ™ci przypadajÄ…cej na jeden rdzeĹ„ obliczeniowy (domyĹ›lnie 5GB na rdzeĹ„)
#SBATCH --mem=6GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri3-gpu
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gpus-per-task=1
## Plik ze standardowym wyjĹ›ciem
#SBATCH --output="logs/training/st/dropout0.3-lr0.001-bilinear.out"

echo "JOB_ID"
echo $SLURM_JOB_ID

cd $HOME/repos/fl-varying-normalization

source ./configs/config.sh

run_with_args exe.trainings.classical_train