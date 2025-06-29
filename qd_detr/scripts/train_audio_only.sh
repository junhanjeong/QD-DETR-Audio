#!/bin/bash

dset_name=hl
ctx_mode=audio_tef  # changed from video_tef to audio_tef
v_feat_types=""  # no video features
t_feat_type=clip
a_feat_type=pann
results_root=results
exp_id=audio_only_exp

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup audio+text features
feat_root=../features

# no video features for audio-only training
v_feat_dim=0
v_feat_dirs=()

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/umt_clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# audio features
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/umt_pann_features/
  a_feat_dim=2050
else
  echo "Wrong arg for a_feat_type."
  exit 1
fi

#### training
bsz=32

echo "Training audio-only model..."
echo "Context mode: ${ctx_mode}"
echo "Audio features: ${a_feat_type} (dim: ${a_feat_dim})"
echo "Text features: ${t_feat_type} (dim: ${t_feat_dim})"
echo "Experiment ID: ${exp_id}"

PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
