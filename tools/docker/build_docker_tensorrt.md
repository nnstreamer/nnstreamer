# host setup
```
cd ${HOME}/src/projects
git clone https://github.com/nnstreamer/nnstreamer.git
git clone https://github.com/nnstreamer/nnstreamer-example.git
cd nnstreamer
```

# Build and run container
```
# Execute from nnstreamer repository root (i.e. ${HOME}/src/projects/nnstreamer)
docker build -f tools/docker/Dockerfile.tensorrt --target=cuda-trt10-base -t cuda-trt10-base:latest --progress=plain .
# Enable docker container to connect to host X server
xhost+
# Run docker container, so we can manually build nnstreamer[-example] and test
docker run \
    --name cuda-trt10-base \
    --rm --interactive --tty --entrypoint /bin/bash \
    --privileged --cap-add SYSLOG --gpus all --env DISPLAY=${DISPLAY} --env NVIDIA_DRIVER_CAPABILITIES=all \
    --mount type=bind,src=/tmp/.X11-unix/,dst=/tmp/.X11-unix \
    --mount type=bind,src=${HOME}/src/projects,dst=/mnt/projects \
    --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
    --workdir /mnt/projects \
    --network host \
    cuda-trt10-base:latest
```

# Build nnstreamer in container

THE FOLLOWING STEPS ARE EXECUTED FROM WITHIN THE CONTAINER

```
# Verify that the gstreamer elements can be found
gst-inspect-1.0 ximagesink
```

```
# Based on: Documentation/how-to-run-examples.md
# Configure the build for nnstreamer
cd /mnt/projects/nnstreamer
export NNST_INSTALLDIR=/mnt/projects/nnstreamer/install
meson --wipe --prefix=${NNST_INSTALLDIR} build
```

Verify the output contains the following:
```
...
Run-time dependency cuda-12.1 found: YES 12.1
Run-time dependency cudart-12.1 found: YES 12.1
Library nvinfer found: YES
...
```

```
# Build and install
ninja -C build install
```

```
# Verify pkg-config for nnstreamer
export PKG_CONFIG_PATH=$NNST_INSTALLDIR/lib/pkgconfig
pkg-config --debug nnstreamer
```

```
# Verify local installation
export GST_PLUGIN_PATH=/mnt/projects/nnstreamer/build/gst/nnstreamer
gst-inspect-1.0 nnstreamer
```

# Build nnstreamer-example in container

```
cd /mnt/projects/nnstreamer-example

# Configure the build for nnstreamer-example
meson --wipe --prefix=${NNST_INSTALLDIR} build
```

Verify the output contains the following:
```
...
Run-time dependency nnstreamer found: YES 2.4.1
...
```

```
# Build and install
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$NNST_INSTALLDIR/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$NNST_INSTALLDIR/include
ninja -C build install
```

```
# Download tflite model to test if nnstreamer works
mkdir models && cd models
/mnt/projects/nnstreamer-example/bash_script/example_models/get-model.sh object-detection-tflite
```

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NNST_INSTALLDIR/lib
/mnt/projects/nnstreamer-example/bash_script/example_object_detection_tensorrt/gst-launch-object-detection-tensorrt.sh
```
