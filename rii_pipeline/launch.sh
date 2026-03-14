#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Robot Inclusivity Index (RII) — Launch Script
# Usage:  ./launch.sh
# ──────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source ROS if available (needed for shell workers)
for distro in "${ROS_DISTRO:-}" jazzy humble iron rolling; do
    if [ -n "${distro:-}" ] && [ -f "/opt/ros/${distro}/setup.bash" ]; then
        source "/opt/ros/${distro}/setup.bash"
        break
    fi
done

cd "${SCRIPT_DIR}"
exec python3 rii_pipeline.py "$@"
