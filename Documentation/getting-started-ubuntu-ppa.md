## Getting Started: Ubuntu-PPA Install


### Install via PPA repository (Ubuntu)

The nnstreamer releases are at a PPA repository. In order to install it, use:

```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install nnstreamer
```

### Additional plugins available

* nnstreamer-caffe2 : Allows to use caffe2 models in a pipeline. (From pytorch 1.3.1 by default)
* nnstreamer-cpp : Allows to use C++ classes as filters of a pipeline.
* nnstreamer-cpp-dev : Required to build C++ class-filters.
* nnstreamer-dev : Required to build C function-filters and to build your own nnstreamer plugins.
* nnstreamer-edgetpu : Allows to use edge-TPU in a pipeline.
* nnstreamer-flatbuf : Allows to convert-from and decode-to flatbuf streams.
* nnstreamer-openvino : Allows to use OpenVINO (Intel), enabling Movidius-X.
* nnstreamer-protobuf : Allows to convert-from and decode-to protobuf streams.
* nnstreamer-python2 : Allows to use python2 classes as filters of a pipeline.
* nnstreamer-python3 : Allows to use python3 classes as filters of a pipeline.
* nnstreamer-pytorch : Allows to use Pytorch models in a pipeline. (From pytorch 1.3.1 by default)
* nnstreamer-tensorflow : Allows to use TensorFlow models in a pipeline. (From tensorflow 1.13.1 by default)
* nnstreamer-tensorflow-lite : Allows to use TensorFlow-lite models in a pipeline. (From tensorflow 1.13.1 by default)


### If you want to use different versions of TensorFlow or PyTorch

#### Safe method (need rebuild)

You need to rebuild nnstreamer's corresponding subplugins (e.g., nnstreamer-tensorflow) with the nerual network framework version you want to use.

* You may configure/update, build with pdebuild/debuild, and install its resulting .deb packages [Ubuntu: Pbuilder / Pdebuild](./getting-started-ubuntu-debuild.md).
* You may configure/update, build with meson/ninja, and install binraies with ninja [Linux generic: build with meson and ninja](./getting-started-meson-build.md): For advanced users with feature customization.
    * Be careful on install paths and duplicated installation. You need to check the configuration (/etc/nnstreamer.ini and env-vars)

#### Unsafe method (no need for rebuild)

Try to let prebuilt nnstreamer binraies use another versions of tensorflow/pytorch installed. Theoretically, it should work by simply replacing tensorflow/pytorch with different versions. Unless symbols and their semantics are chnaged, it should work. (but that happens often with neural network frameworks, which are still not that stable.)
