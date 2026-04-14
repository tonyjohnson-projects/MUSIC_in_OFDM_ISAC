#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PID="$$"
PARENT_PID="${PPID:-}"

collect_python_pids() {
  for pid in $(pgrep -x Python 2>/dev/null || true); do
    if [[ "$pid" == "$SELF_PID" || -n "$PARENT_PID" && "$pid" == "$PARENT_PID" ]]; then
      continue
    fi
    if lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | grep -Fq "$ROOT_DIR"; then
      echo "$pid"
    fi
  done
}

collect_shell_wrapper_pids() {
  local pattern='run_study\.py|build_submission_bundle\.sh|build_results_bundle\.sh|run_staged_submission\.py|run_model_order_comparison_64trials\.py'
  ps -axo pid=,command= | while read -r pid command; do
    if [[ "$pid" == "$SELF_PID" || -n "$PARENT_PID" && "$pid" == "$PARENT_PID" ]]; then
      continue
    fi
    if [[ "$command" =~ $pattern ]] && [[ "$command" == *"$ROOT_DIR"* ]]; then
      echo "$pid"
    fi
  done
}

TARGET_PIDS=()
while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  TARGET_PIDS+=("$pid")
done < <(
  {
    collect_python_pids
    collect_shell_wrapper_pids
  } | awk 'NF {print $1}' | sort -n | uniq
)

if [[ "${#TARGET_PIDS[@]}" -eq 0 ]]; then
  echo "No project run processes found under $ROOT_DIR"
  exit 0
fi

echo "Stopping project run processes from $ROOT_DIR"
for pid in "${TARGET_PIDS[@]}"; do
  ps -p "$pid" -o pid=,ppid=,rss=,etime=,command=
done

echo
echo "Sending SIGTERM..."
/bin/kill "${TARGET_PIDS[@]}" 2>/dev/null || true
sleep 2

SURVIVORS=()
for pid in "${TARGET_PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    SURVIVORS+=("$pid")
  fi
done

if [[ "${#SURVIVORS[@]}" -gt 0 ]]; then
  echo "Force killing survivors with SIGKILL: ${SURVIVORS[*]}"
  /bin/kill -9 "${SURVIVORS[@]}" 2>/dev/null || true
  sleep 1
fi

echo
echo "Remaining matching processes:"
remaining=0
for pid in $(pgrep -x Python 2>/dev/null || true); do
  if lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | grep -Fq "$ROOT_DIR"; then
    ps -p "$pid" -o pid=,ppid=,rss=,etime=,command=
    remaining=1
  fi
done
if [[ "$remaining" -eq 0 ]]; then
  echo "None"
fi
