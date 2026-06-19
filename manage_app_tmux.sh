#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-nsarxiv-app}"
ENV_NAME="${ENV_NAME:-nsarxiv-app}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-8501}"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|status|attach|logs}

Commands:
  start    Start Streamlit in tmux session '$SESSION_NAME'
  stop     Stop tmux session '$SESSION_NAME'
  restart  Restart tmux session '$SESSION_NAME'
  status   Show tmux and HTTP status
  attach   Attach to tmux session '$SESSION_NAME'
  logs     Show recent tmux pane output
EOF
}

has_session() {
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
}

port_pid() {
    ss -ltnp "( sport = :$PORT )" 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -n1
}

start_app() {
    if has_session; then
        echo "tmux session '$SESSION_NAME' is already running"
        return 0
    fi

    local pid
    pid="$(port_pid)"
    if [ -n "$pid" ]; then
        echo "port $PORT is already in use by PID $pid"
        echo "stop that process first or run '$0 status' to inspect"
        return 1
    fi

    tmux new-session -d -s "$SESSION_NAME" \
        "bash -lc 'cd \"$SCRIPT_DIR\" && if [ -f .env ]; then set -a && source ./.env && set +a; fi && source \"$CONDA_SH\" && conda activate \"$ENV_NAME\" && exec streamlit run main.py --server.headless true --server.address 127.0.0.1 --server.port \"$PORT\" --browser.gatherUsageStats false --server.enableStaticServing true'"

    echo "started tmux session '$SESSION_NAME'"
    sleep 3
    "$0" status
}

stop_app() {
    if ! has_session; then
        echo "tmux session '$SESSION_NAME' is not running"
        return 0
    fi
    tmux kill-session -t "$SESSION_NAME"
    echo "stopped tmux session '$SESSION_NAME'"
}

status_app() {
    if has_session; then
        echo "tmux: running ($SESSION_NAME)"
        tmux list-panes -t "$SESSION_NAME" -F 'pane #{pane_index}: #{pane_current_command}'
    else
        echo "tmux: not running ($SESSION_NAME)"
    fi

    local pid
    pid="$(port_pid)"
    if [ -n "$pid" ]; then
        echo "port $PORT: listening (PID $pid)"
        curl -fsS "http://127.0.0.1:$PORT/" >/dev/null && echo "http://127.0.0.1:$PORT: reachable" || true
    else
        echo "port $PORT: not listening"
    fi
}

attach_app() {
    if ! has_session; then
        echo "tmux session '$SESSION_NAME' is not running"
        return 1
    fi
    exec tmux attach -t "$SESSION_NAME"
}

show_logs() {
    if ! has_session; then
        echo "tmux session '$SESSION_NAME' is not running"
        return 1
    fi
    tmux capture-pane -pt "$SESSION_NAME":0 -S -80
}

command="${1:-status}"

case "$command" in
    start) start_app ;;
    stop) stop_app ;;
    restart) stop_app; start_app ;;
    status) status_app ;;
    attach) attach_app ;;
    logs) show_logs ;;
    *) usage; exit 1 ;;
esac
