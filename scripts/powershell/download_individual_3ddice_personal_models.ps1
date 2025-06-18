$normalization_names=@('fcm', 'minmax', 'nonorm', 'nyul', 'whitestripe', 'zscore')
$model_dir='model-fedbn-lr0.001-rd32-ep2-2025-05-22'


foreach ($normalization_name in $normalization_names) {
    mkdir $model_dir/client_$normalization_name
    scp "plgfiszer@ares.cyfronet.pl:/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/trained_models/$model_dir/FedBN(batch_norm=LayerNormalizationType.GN)_client_$normalization_name/best_model.pth" $model_dir/client_$normalization_name/best_model.pth
}