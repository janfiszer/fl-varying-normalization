$normalization_names=@('fcm', 'minmax', 'nonorm', 'nyul', 'whitestripe', 'zscore')
$model_dir='model-feddelaybn-lr0.001-rd32-ep2-2025-05-26'

mkdir trained_models/$model_dir/3d_dice
foreach ($normalization_name in $normalization_names) {
  scp "plgfiszer@ares.cyfronet.pl:/net/pr2/projects/plgrid/plggflmri/Data/Internship/FL/varying-normalization/trained_models/$model_dir/FedDelay_client_$normalization_name/preds_from_nonorm/$normalization_name/*.pkl" model-fed-bndelay-lr0.001-rd32-ep2-2025-05-26/3d_dice
}