#!/bin/bash -l

## Nazwa zlecenia
#SBATCH -J FedBN
# Liczba alokowanych wÄ™zĹ‚Ăłw
#SBATCH -n 7
## IloĹ›Ä‡ pamiÄ™ci przypadajÄ…cej na jeden rdzeĹ„ obliczeniowy (domyĹ›lnie 5GB na rdzeĹ„)
##SBATCH --mem=100GB
#SBATCH --mem-per-cpu=10GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=02:00:00
## Nazwa grantu do rozliczenia zuĹĽycia zasobĂłw
#SBATCH -A plgfmri3-gpu-a100
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gpus-per-task=1
## Plik ze standardowym wyjĹ›ciem
#SBATCH --output="logs/trainings/fl/fedbn/parent.out"

cd $HOME/repos/fl-varying-normalization

DIR_NAME=logs/trainings/fl/fedbn
echo $DIR_NAME

normalization_names=("fcm" "minmax" "nonorm" "nyul" "whitestripe" "zscore")

PORT=8088

## All the sruns are launched on the same node.
echo Server starting...
srun --ntasks=1 --cpus-per-task=1 --output="./$DIR_NAME/server.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python -m exe.trainings.run_server $PORT fedbn &

echo "Sleep for 300 seconds"
sleep 300

for norm_name in "${normalization_names[@]}"; do
    echo "Client '$norm_name' starting..."
    srun --ntasks=1 --cpus-per-task=1 --output="./$DIR_NAME/$norm_name.out" \
    $PLG_GROUPS_STORAGE/plggflmri/anaconda3/bin/python -m exe.trainings.run_client_train \
    /net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/data/$norm_name $norm_name $PORT fedbn &
done
wait

echo The end, not to be continued...