#!/bin/bash -l


## Nazwa zlecenia
#SBATCH -J "3ddice"
# Liczba alokowanych wÄ™zĹ‚Ăłw
#SBATCH -N 1
## Liczba zadaĹ„ per wÄ™zeĹ‚ (domyĹ›lnie jest to liczba alokowanych rdzeni na wÄ™Ĺşle)
#SBATCH --ntasks-per-node=1
## IloĹ›Ä‡ pamiÄ™ci przypadajÄ…cej na jeden rdzeĹ„ obliczeniowy (domyĹ›lnie 5GB na rdzeĹ„)
#SBATCH --mem-per-cpu=4GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=01:30:00
## Nazwa grantu do rozliczenia zuĹĽycia zasobĂłw
#SBATCH -A plgfmri3-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjĹ›ciem
#SBATCH --output="logs/evaluations/both/from_nonorm/3d_dice/paren.out"

# Define the models list
# models=(
#     "model-whitestripe-ep16-lr0.001-GN-2025-04-30-13h" "model-nonorm-ep16-lr0.001-GN-2025-04-30-13h" "model-minmax-ep16-lr0.001-GN-2025-04-30-13h" "model-fcm-ep16-lr0.001-GN-2025-04-30-13h" "model-nyul-ep16-lr0.001-GN-2025-04-30-13h" "model-zscore-ep16-lr0.001-GN-2025-04-29-13h" "model-all-ep16-lr0.001-GN-2025-04-30-13h" "model-fedavg-lr0.001-rd32-ep2-2025-05-08"
# )
models=(
    "model-nonorm-ep16-lr0.001-GN-2025-06-05-12h"
)

normalization_names=("fcm" "minmax" "nonorm" "nyul" "whitestripe" "zscore")
# normalization_names=("zscore")


# Iterate over models and directories
for normalization_name in "${normalization_names[@]}"; do
    echo $normalization_name
    for model_dir in "${models[@]}"; do
        echo "Evaluating model: $model_dir"

        srun --output="logs/evaluations/both/from_nonorm/3d_dice/$normalization_name-data/$model_dir.out" $PLG_GROUPS_STORAGE/plggflmri/new_conda/fl/bin/python -u -m exe.evaluations.compute_3d \
        /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/data/nonorm/test /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/preds_from_nonorm/$normalization_name/test
    done
done
