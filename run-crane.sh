#!/bin/bash
xhost +local:docker 2>/dev/null
xhost +si:localuser:root 2>/dev/null

mkdir -p ~/docker/isaac-sim/cache/{ov,pip,glcache,computecache}

docker run --gpus all --rm -it \
  --name isaac-sim \
  --entrypoint bash \
  -e DISPLAY=$DISPLAY \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e XAUTHORITY=/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v ~/isaac-sim/scripts:/workspace/scripts \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  --network host --privileged \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  -c "cd /isaac-sim && ./isaac-sim.sh --exec '/workspace/scripts/crane_script_editor.py'"
