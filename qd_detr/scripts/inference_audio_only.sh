#!/bin/bash

ckpt_path=$1
eval_split_name=$2
a_feat_type=pann
a_feat_dim=2050
feat_root=../features
a_feat_dir=${feat_root}/umt_pann_features/
t_feat_dir=${feat_root}/umt_clip_text_features/
eval_path=data/highlight_${eval_split_name}_release.jsonl

echo "Running audio-only inference..."
echo "Checkpoint: ${ckpt_path}"
echo "Eval split: ${eval_split_name}"
echo "Audio features: ${a_feat_dir}"
echo "Text features: ${t_feat_dir}"

PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--v_feat_dim 0 \
${@:3}
