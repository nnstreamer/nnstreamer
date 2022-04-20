---
title: Writing tensor filter subplugin
...

# Writing a subplugin for NNStreamer's tensor\_filter

You can support a new neural-network framework (e.g., Tensorflow and Caffe) or a new neural-network hardware accelerator with its own software interface (e.g., openVINO for NCS and some python library for edge-TPU) by writing a tensor\_filter subplugin.

It is called "**subplugin**" because it is a plugin for a GStreamer plugin, ```tensor_filter```.



## Quick guide on writing a tensor\_filter subplugin for a new framework/hardware.

You can start writing a ```tensor_filter``` subplugin easily by using code-template/generator from nnstreamer-example.git. It is in ```/templates/tensor_filter_subplugin``` of ```nnstreamer-example.git```. The following is how to start writing a subplugin with the template for Tizen devices (5.5 M2 +). In this example, the target subplugin name is ```example```.

```
$ git clone https://github.com/nnstreamer/nnstreamer-example.git
...
$ cd nnstreamer-example
$ cd templates/tensor_filter_subplugin
$ ./deploy example ~/example
...
...
$ cd ~/example
$ ls -l
total 124
-rw-rw-r--   1 babot babot    697 10-11 23:57 meson.build
drwxrwxr-x   2 babot babot   4096 10-11 23:57 packaging
drwxrwxr-x   2 babot babot   4096 10-11 23:57 src
$ ls -l src/
total 4
-rw-rw-r-- 1 babot babot 3960 10-11 23:57 tensor_filter_subplugin.c
$
```

Then, in ```src``` directory, you can fill in the callbacks.

If you need to add dependencies for additional libraries of your own libraries, edit meson.build and packaging/*.spec accordingly as well.

Then, use ```gbs``` to build and package your subplugin for Tizen:

```
$ gbs build
...
```

Although we supply a packaging script for Tizen only, the code and build script (meson.build) supports other software platforms as well; you may build it with meson and install to appropriate paths.


## More about the template subplugin code

In case you are interested in the internals, here goes a few more details.

### License

NNStreamer plugins and libraries are licensed as LGPL. Thus subplugins and applications based on NNStreamer may be licensed with any licenses including proprietary licenses (non-open source) as long as NNStreamer is used as shared libraries assuming that other LGPL conditions are met. Besides, we allow to write subplugins with the given template codes without any licensing condition. Thus, do not worry about licensing for subplugins.


### Dependencies / Libraries

As you can see in packaging/*.spec and meson.build of the template, ```nnstreamer-dev``` is the only mandatory dependency. Anyway, of course, you need to add dependencies for your own library/hardware usages.

In order to provide callbacks required by ```tensor_filter```, you need to include ```nnstreamer_plugin_api_filter.h```, which is supplied with ```nnstreamer-dev``` package (in Tizen or Ubuntu).

### Testing

There is a templated test suite provided inside `nnstreamer-test-dev` package for dpkg, `nnstreamer-test-devel` for tizen distro.
You may install or _BuildRequire_ this package to utilize predefined test templates.
After installing the package, you can locate the package.


```sh
$ TEMPLATE_DIR=$(pkg-config nnstreamer --variable=test_templatedir)
$ echo $TEMPLATE_DIR # result varies depending on the platform
/usr/lib/nnstreamer/bin/unittest-nnstreamer
$ ls $TEMPLATE_DIR
nnstreamer-test.ini.in  subplugin_unittest_template.cc.in
```

There are two test template provided. One is conf file for the test environment, and the other is for the basic unittests (requires gtest).

List of variables that needs to be provided are provided below...
| File                              | Variables                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| nnstreamer-test.ini.in            | SUBPLUGIN_INSTALL_PREFIX, ENABLE_ENV_VAR, ENABLE_SYMBOLIC_LINK, TORCH_USE_GPU, TFLITE_SUBPLUGIN_PRIORITY, ELEMENT_RESTRICTION_CONFIG |
| subplugin_unittest_template.cc.in | EXT_NAME, EXT_ABBRV, MODEL_FILE,                                                                                                     |

You can change the provided each variable on the go using a script, for example...
```sh
$ sed s/@EXT_ABBRV@/cutom_plugin/g subplugin_unittest_template.cc.in > subplugin_unittest.cc
```

but preferably, using meson through `configure_data` and `configure_file`

and defining `NNSTREAMER_CONF_PATH` environment variable will locate the `nnstreamer-test.ini` to find your subplugin

with generated `nnstreamer-test.ini`

You can run the test as below:

```sh
NNSTREMAER_CONF_PATH=your/conf/file.ini ./subplugin_unittest
```

### Install Path

The default ```tensor_filter``` subplugin path is ```/usr/lib/nnstreamer/filters/```. It can be modified by configuring ```/etc/nnstreamer.ini```.
### Flexible/Dynamic Input/Output Tensor Dimension

Although the given template code supports static input/output tensor dimension (a single neural network model is supposed to have a single set of input/output tensor/tensors dimensions), NNStreamer's ```tensor_filter``` itself supports dynamic input/output tensor/tensors dimensions; output dimensions may be determined by input dimensions, which is determined at run-time.

In order to support this, you need to supply an additional callback, ```setInputDimension``` defined in ```GstTensorFilterFramework``` of ```nnstreamer_plugin_api_filter.h```.


### Writing one from scratch

In normal usage cases, a subplugin exists as a shared library loaded dynamically (dlopen) by yet another shared library, ```tensor_filter```. By registering a ```tensor_filter``` object (a struct instance of GstTensorFilterFramework) with an init function, NNStreamer recognizes it with the given name. In the template code, it is registered with a function ```init_filter_${name}()```. For more information, refer to the doxygen entries of GstTensorFilterFramework in the header file.

For more information about the struct, refer to [Doxygen Doc on GstTensorFilterFramework](http://ci.nnstreamer.ai/nnstreamer/html/struct__GstTensorFilterFramework.html).
