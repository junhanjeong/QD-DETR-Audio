#!/bin/bash

# UMT-style QD-DETR-Audio Training Script
dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
a_feat_type=pann
results_root=results
exp_id=umt_custom

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text+audio features
feat_root=../features

# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_features)
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_text_features/
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

# audio features
if [[ ${a_feat_type} == "pann" ]]; then
  a_feat_dir=${feat_root}/pann_features/
  a_feat_dim=2050
else
  echo "Wrong arg for a_feat_type."
  exit 1
fi

#### UMT-specific parameters
umt_hidden_dim=256
umt_num_tokens=4
umt_num_layers=1

#### training
bsz=32
n_epoch=200
lr_drop=100

# model config
num_queries=10
enc_layers=2
dec_layers=2

# loss config
span_loss_coef=10
giou_loss_coef=1
label_loss_coef=4
lw_saliency=1

PYTHONPATH=$PYTHONPATH:. python qd_detr/train_umt.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--umt_hidden_dim ${umt_hidden_dim} \
--umt_num_tokens ${umt_num_tokens} \
--umt_num_layers ${umt_num_layers} \
--bsz ${bsz} \
--n_epoch ${n_epoch} \
--lr_drop ${lr_drop} \
--num_queries ${num_queries} \
--enc_layers ${enc_layers} \
--dec_layers ${dec_layers} \
--span_loss_coef ${span_loss_coef} \
--giou_loss_coef ${giou_loss_coef} \
--label_loss_coef ${label_loss_coef} \
--lw_saliency ${lw_saliency} \
--results_root ${results_root} \
--exp_id ${exp_id} \
${@:1}
