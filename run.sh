#!/bin/bash
# Isaac Sim Docker 啟動腳本
# 用法: bash run.sh [headless|streaming]

ISAAC_SIM_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# X11 授權
xhost +local:docker 2>/dev/null
xhost +si:localuser:root 2>/dev/null

# Cache 目錄（加速後續啟動）
mkdir -p ~/docker/isaac-sim/cache/{ov,pip,glcache,computecache,asset_browser}
mkdir -p ~/docker/isaac-sim/{logs,data,pkg,documents}

# 共用參數
COMMON_ARGS=(
    --gpus all
    --rm -it
    --entrypoint bash
    -e DISPLAY=$DISPLAY
    -e ACCEPT_EULA=Y
    -e XAUTHORITY=/root/.Xauthority
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -e NVIDIA_VISIBLE_DEVICES=all
    -e OMNI_KIT_ALLOW_ROOT=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v "$HOME/.Xauthority:/root/.Xauthority:ro"
    -v "$PROJECT_DIR/scripts:/workspace/scripts"
    -v "$PROJECT_DIR/assets:/workspace/assets"
    -v "$PROJECT_DIR/logs:/workspace/logs"
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw
    --network host
    --privileged
    --name isaac-sim
)

MODE="${1:-gui}"

case "$MODE" in
    headless)
        echo "啟動 Isaac Sim (headless 模式)..."
        docker run "${COMMON_ARGS[@]}" \
            "$ISAAC_SIM_IMAGE" \
            -c "/isaac-sim/isaac-sim.sh --no-window"
        ;;
    streaming)
        echo "啟動 Isaac Sim (streaming 模式)..."
        docker run "${COMMON_ARGS[@]}" \
            "$ISAAC_SIM_IMAGE" \
            -c "/isaac-sim/isaac-sim.streaming.sh"
        ;;
    *)
        echo "啟動 Isaac Sim (GUI 模式)..."
        docker run "${COMMON_ARGS[@]}" \
            "$ISAAC_SIM_IMAGE" \
            -c "/isaac-sim/isaac-sim.sh"
        ;;
esac
