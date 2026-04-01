#!/bin/bash
# tmux development session for BFM-Zero deployment
# Layout:
#   Window 0 "sim":    Sim-to-Sim testing (2 panes: MuJoCo + Policy)
#   Window 1 "claude": Claude Code
#   Window 2 "data":   Data conversion (2 panes: script + ~/my_data)
#   Window 3 "config": Config editing (tracking / reward / goal / model files)

SESSION="bfm"
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# ─── Window 0: sim (S2S simulation testing) ──────────────────────────────────
tmux new-session -d -s "$SESSION" -c "$WORK_DIR"
tmux rename-window -t "$SESSION" "sim"

# Split into 2 panes: left (MuJoCo) + right (Policy)
tmux split-window -t "$SESSION:0.0" -h -c "$WORK_DIR"

# Pane 0: MuJoCo Simulator
tmux send-keys -t "$SESSION:0.0" "conda activate bfm0real" Enter
tmux send-keys -t "$SESSION:0.0" "# MuJoCo Simulator" Enter
tmux send-keys -t "$SESSION:0.0" "# Run: python -m sim_env.base_sim --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof.yaml" Enter

# Pane 1: Policy Deploy
tmux send-keys -t "$SESSION:0.1" "conda activate bfm0real" Enter
tmux send-keys -t "$SESSION:0.1" "# Policy Deploy" Enter
tmux send-keys -t "$SESSION:0.1" "# Run: ./rl_policy/tracking.sh  (or reward.sh / goal.sh)" Enter

# ─── Window 1: claude ────────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "claude" -c "$WORK_DIR"
tmux send-keys -t "$SESSION:1" "claude" Enter

# ─── Window 2: data (Data conversion) ────────────────────────────────────────
tmux new-window -t "$SESSION" -n "data" -c "$WORK_DIR"

# Split into 2 panes: left (script) + right (~/my_data)
tmux split-window -t "$SESSION:2.0" -h -c "$HOME/my_data"

# Pane 0: Data conversion script
tmux send-keys -t "$SESSION:2.0" "conda activate bfm0inf" Enter
tmux send-keys -t "$SESSION:2.0" "python scripts/motion_to_tracking_z.py -h" Enter

# Pane 1: ~/my_data directory listing
tmux send-keys -t "$SESSION:2.1" "ls" Enter

# ─── Window 3: config ────────────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "config" -c "$WORK_DIR"

# Split into 4 panes: 2x2 grid
tmux split-window -t "$SESSION:3.0" -v -c "$WORK_DIR"
tmux split-window -t "$SESSION:3.0" -h -c "$WORK_DIR"
tmux split-window -t "$SESSION:3.2" -h -c "$WORK_DIR"

# Pane 0: tracking config (top-left)
tmux send-keys -t "$SESSION:3.0" "nano config/exp/tracking/walking.yaml" Enter

# Pane 1: reward config (top-right)
tmux send-keys -t "$SESSION:3.1" "nano config/exp/reward/locomotion.yaml" Enter

# Pane 2: goal config (bottom-left)
tmux send-keys -t "$SESSION:3.2" "nano config/exp/goal/goal.yaml" Enter

# Pane 3: model files (bottom-right)
tmux send-keys -t "$SESSION:3.3" "cd model && ls -R" Enter

# Focus on sim window
tmux select-window -t "$SESSION:0"
tmux select-pane -t "$SESSION:0.0"

# Attach
tmux attach-session -t "$SESSION"
