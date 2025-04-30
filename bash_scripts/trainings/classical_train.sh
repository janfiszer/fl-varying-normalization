#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J SngTrain
# Liczba alokowanych wÄ™zĹ‚Ăłw
#SBATCH -n 1
## Liczba zadaĹ„ per wÄ™zeĹ‚ (domyĹ›lnie jest to liczba alokowanych rdzeni na wÄ™Ĺşle)
#SBATCH --ntasks-per-node=1
## IloĹ›Ä‡ pamiÄ™ci przypadajÄ…cej na jeden rdzeĹ„ obliczeniowy (domyĹ›lnie 5GB na rdzeĹ„)
#SBATCH --mem=8GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:00:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgfmri3-gpu-a100
#SBATCH --gres=gpu:1
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
## Plik ze standardowym wyjĹ›ciem
##SBATCH --output="logs/trainings/st/layer_norm_test_more_metrics/zscoreNoNorm.out"
#SBATCH --error="logs/trainings/st/layer_norm_test_more_metrics/zscoreNoNorm.err"

cd $HOME/repos/fl-varying-normalization

srun $PLG_GROUPS_STORAGE/plggflmri/new_conda/fl/bin/python -u -m exe.trainings.classical_train \
    /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/data/zscore