#!/usr/bin/env bash
# Online tracking: B-network computes z per frame from a motion source.
# Toggle file/zmq via config/exp/tracking_online/walking.yaml.

POLICY_CONFIG=config/policy/motivo_newG1.yaml
MODEL_ONNX_PATH=./model/exported/FBcprAuxModel_policy_test.onnx
TASK=config/exp/tracking_online/walking.yaml

python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1.yaml \
    --policy_config ${POLICY_CONFIG} \
    --model_path ${MODEL_ONNX_PATH} \
    --task  ${TASK}
