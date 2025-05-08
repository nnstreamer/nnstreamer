---
title: Writing tensor filter subplugin
...

# Writing a subplugin for NNStreamer's tensor\_filter

You can support a new neural-network framework (e.g., Tensorflow and Caffe) or a new neural-network hardware accelerator with its own software interface (e.g., openVINO for NCS and some python library for edge-TPU) by writing a tensor\_filter subplugin.

It is called "**subplugin**" because it is a plugin for a GStreamer plugin, ```tensor_filter```.

**A subplugin should NEVER require properties of input/output dimensions or types.** You should use such properties for pipeline debugging and validation and the subplugin should fetch such information directly from model files via framework APIs and GSTCAPS negotiations.



# Quick guide on writing a tensor\_filter subplugin for a new framework/hardware.

The most recommended method is to write a C++ tensor\_filter subplugin as a derived class of ```tensor_filter_subplugin``` class [/gst/nnstreamer/include/nnstreamer_cppplugin_api_filter.hh].
Then, you can register the derived class, usually by calling ```tensor_filter_subplugin::register_subplugin()``` in the init function so that subplugin infrastructure recognizes your subplugin when the shared library is loaded.


## A tensor\_filter subplugin C++ class

- Interface: [/gst/nnstreamer/include/nnstreamer_cppplugin_api_filter.hh](../gst/nnstreamer/include/nnstreamer_cppplugin_api_filter.hh)
- Reference (example): [/ext/nnstreamer/tensor_filter/tensor_filter_snap.cc](../ext/nnstreamer/tensor_filter/tensor_filter_snap.cc)

You may also find other subplugins inheriting the ```tensor_filter_subplugin``` class in [/ext/nnstreamer/tensor_filter] that can be used as examples.


If the framework or backend/runtime library has C++ APIs or you want to write the subplugin in C++, use ```#include <nnstreamer_cppplugin_api_filter.hh>``` and inherit the base class, ```nnstreamer::tensor_filter_subplugin```.
With this interface, subplugin writers are supposed to write their own concrete class based on ```nnstreamer::tensor_filter_subplugin``` by filling up virtual methods (methods marked ```These should be filled/implemented by subplugin authores```).
Mandatory methods to be filled are declared as pure virtual function and optional methods are declared as regular virtual function (```eventHandler```).


As in C subplugin, the derived (concrete) class should be registered at init and unregistered at exit.
Subplugin writers are supposed to use the static methods of the base class, ```register_subplugin()``` and ```unregister_subplugin()```; refer to the function ```init_filter_snap()``` and ```fini_filter_snap()``` in the reference example.



Note that C++ subplugin is simpler and easy-to-maintain compared to C subplugin. Unless you really need to write your subplugin in C, we recommend to use the ```nnstreamer::tensor_filter_subplugin``` base class.



## A tensor\_filter subplugin in C

- Interface: [/gst/nnstreamer/include/nnstreamer_plugin_api_filter.h](../gst/nnstreamer/include/nnstreamer_plugin_api_filter.h)
- Reference (example): [/ext/nnstreamer/tensor_filter/tensor_filter_nnfw.c](../ext/nnstreamer/tensor_filter/tensor_filter_nnfw.c)


If the framework or backend/runtime library has C APIs and you want to write the subplugin in C, use ```#include <nnstreamer_plugin_api_filter.h>```.
Your C subplugin is supposed to fill in ```GstTensorFilterFramework``` struct and register the struct with ```nnstreamer_filter_probe (GstTensorFilterFrameworkEventData *)``` function, which is supposed to be called with ```((constructor))``` initializer (```init_filter_nnfw (void)``` function in the reference).
If your subplugin has custom properties to be supplied by users, describe their usages with ```nnstreamer_filter_set_custom_property_desc ()``` function.
Then, call ```nnstreamer_filter_exit ()``` function with ```((destructor))``` terminator (```fini_filter_nnfw (void)``` function in the reference).


In ```GstTensorFilterFramework```, there are two different ways, ```v0 (version == GST_TENSOR_FILTER_FRAMEWORK_V0)``` and ```v1 (version == GST_TENSOR_FILTER_FRAMEWORK_V1)```. In the struct, there is a ```union``` of ```v0``` and ```v1```, and it is recommended to use ```v1``` and ```set version = GST_TENSOR_FILTER_FRAMEWORK_V1``` (v1). ```v0``` is supposed to be used by old subplugins for backward compatibility and any new subplugins should use ```v1```, which is simpler and richer in features.


However, note that if you are going to use framework/library with C++ APIs, please do not use ```nnstreamer_plugin_api_filter.h```, but use the base tensor-filter-subplugin C++ class as described in the previous section.



## Code generator for tensor\_filter subplugin in C

**This generator has not been maintained recently.**

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


# More about the template subplugin code

In case you are interested in the internals, here goes a few more details.

## License

NNStreamer plugins and libraries are licensed as LGPL. Thus subplugins and applications based on NNStreamer may be licensed with any licenses including proprietary licenses (non-open source) as long as NNStreamer is used as shared libraries assuming that other LGPL conditions are met. Besides, we allow to write subplugins with the given template codes without any licensing condition. Thus, do not worry about licensing for subplugins.


## Dependencies / Libraries

As you can see in packaging/*.spec and meson.build of the template, ```nnstreamer-dev``` is the only mandatory dependency. Anyway, of course, you need to add dependencies for your own library/hardware usages.

In order to provide callbacks required by ```tensor_filter```, you need to include ```nnstreamer_plugin_api_filter.h```, which is supplied with ```nnstreamer-dev``` package (in Tizen or Ubuntu).

## Testing

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
$ sed s/@EXT_ABBRV@/custom_plugin/g subplugin_unittest_template.cc.in > subplugin_unittest.cc
```

but preferably, using meson through `configure_data` and `configure_file`

and defining `NNSTREAMER_CONF_PATH` environment variable will locate the `nnstreamer-test.ini` to find your subplugin

with generated `nnstreamer-test.ini`

You can run the test as below:

```sh
NNSTREMAER_CONF_PATH=your/conf/file.ini ./subplugin_unittest
```

## Install Path

The default ```tensor_filter``` subplugin path is ```/usr/lib/nnstreamer/filters/```. It can be modified by configuring ```/etc/nnstreamer.ini```.
## Flexible/Dynamic Input/Output Tensor Dimension

Although the given template code supports static input/output tensor dimension (a single neural network model is supposed to have a single set of input/output tensor/tensors dimensions), NNStreamer's ```tensor_filter``` itself supports dynamic input/output tensor/tensors dimensions; output dimensions may be determined by input dimensions, which is determined at run-time.

In order to support this, you need to supply an additional callback, ```setInputDimension``` defined in ```GstTensorFilterFramework``` of ```nnstreamer_plugin_api_filter.h```.


## Writing one from scratch

In normal usage cases, a subplugin exists as a shared library loaded dynamically (dlopen) by yet another shared library, ```tensor_filter```. By registering a ```tensor_filter``` object (a struct instance of GstTensorFilterFramework) with an init function, NNStreamer recognizes it with the given name. In the template code, it is registered with a function ```init_filter_${name}()```. For more information, refer to the doxygen entries of GstTensorFilterFramework in the header file.

For more information about the struct, refer to [Doxygen Doc on GstTensorFilterFramework](http://ci.nnstreamer.ai/nnstreamer/html/struct__GstTensorFilterFramework.html).
