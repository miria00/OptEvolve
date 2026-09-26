#!/usr/bin/env bash
# Launch a run on the H100 inside a named tmux session, so it survives the ssh
# connection and can be reattached to.  Usage:
#   h100run.sh <session-name> <command...>
# Inspect with:  ssh ... 'tmux ls'  /  'tmux capture-pane -pt <name>'
set -euo pipefail
H="root@38.80.152.148"; P=31233; K="$HOME/.ssh/id_ed25519"
NAME="$1"; shift
ssh -o StrictHostKeyChecking=no -p "$P" -i "$K" "$H" \
  "tmux has-session -t $NAME 2>/dev/null && { echo 'session $NAME already exists'; exit 1; }
   tmux new-session -d -s $NAME '$*'
   sleep 2; tmux ls | grep $NAME"
