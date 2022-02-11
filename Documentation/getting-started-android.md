---
title: Android
...

# NNStreamer API Library for Android

## Note: The API is separated into a [new repository](https://github.com/nnstreamer/api)

## Prerequisite

We assume that you already have experienced Android application developments with Android Studio.

 * Host PC:
   * OS: Ubuntu 16.04 / 18.04 x86_64 LTS
   * Android Studio: Ubuntu version
   * Android SDK: Min version 28 (Pie)
   * Android NDK: Use default ndk-bundle in Android Studio **( < 22.0 )**
   * GStreamer: gstreamer-1.0-android-universal-1.20.0

## Build library

### Environment variables

First of all, you need to set-up the development environment as following:

```bash
$ export ANDROID_DEV_ROOT=$HOME/Android           # Set your own path (default location: $HOME/Android)
$ mkdir -p $ANDROID_DEV_ROOT/tools/sdk
$ mkdir -p $ANDROID_DEV_ROOT/tools/ndk
$ mkdir -p $ANDROID_DEV_ROOT/gstreamer-1.0
$ mkdir -p $ANDROID_DEV_ROOT/workspace
$
$ vi ~/.bashrc
# The environment variables to develop an Android application with NNStreamer
#
export JAVA_HOME=/opt/android-studio/jre            # JRE path in Android Studio
export ANDROID_DEV_ROOT=$HOME/android               # Set your own path (default location: "$HOME/Android".)

# $ANDROID_DEV_ROOT/tools/sdk/: Android SDK root directory (default location: $HOME/Android/Sdk)
# $ANDROID_DEV_ROOT/tools/ndk/: Android NDK root directory (default location: $HOME/Android/Sdk/ndk/<ndk-version>)
# $ANDROID_DEV_ROOT/gstreamer-1.0/: GStreamer binaries
# $ANDROID_DEV_ROOT/workspace/nnstreamer/: The git repository of NNStreamer
# $ANDROID_DEV_ROOT/workspace/api/: The git repository of ML API

export ANDROID_SDK=$ANDROID_DEV_ROOT/tools/sdk
export ANDROID_NDK=$ANDROID_DEV_ROOT/tools/ndk
export ANDROID_SDK_ROOT=$ANDROID_SDK
export ANDROID_NDK_ROOT=$ANDROID_NDK
export GSTREAMER_ROOT_ANDROID=$ANDROID_DEV_ROOT/gstreamer-1.0
export NNSTREAMER_ROOT=$ANDROID_DEV_ROOT/workspace/nnstreamer
export ML_API_ROOT=$ANDROID_DEV_ROOT/workspace/api
```

### Install required packages
Some required packages should be installed as below.
```bash
$ sudo apt install subversion curl pkg-config gradle
```

### Download Android Studio

Download and install Android Studio to compile an Android source code.
You can see the installation guide [here](https://developer.android.com/studio/install).

For example,
```bash
$ firefox  https://developer.android.com/studio &
Then, download the **Android Studio** IDE into the /opt folder as follows.
$ cd /opt
$ sudo curl -O https://r1---sn-n5hn0ob-pjoe.gvt1.com/edgedl/android/studio/ide-zips/3.6.3.0/android-studio-ide-192.6392135-linux.tar.gz
$ sudo tar xvzf ./android-studio-ide-183.5452501-linux.tar.gz
```

Now, run the **Android Studio** IDE as follows.
```bash
$ /opt/android-studio/bin/studio.sh
```
Finally, install SDK into the `$ANDROID_SDK` folder as follows.
The `yes` command automatically agrees to the license question for the Android SDK.
```bash
$ cd $ANDROID_SDK/tools/bin
$ yes | ./sdkmanager --licenses
```

#### Proxy Setting

If your network is maintained a proxy of the office, you need to set-up the proxy and SSL configuration.
* Proxy setting: File > Settings > Appearance & Behavior > System Settings > HTTP Proxy
* SSL Certificate:  File > Settings  > Tools > Server Certificates > Register your certificate and check `Accept non-trusted certificates automatically`


### Download NDK

Use the default NDK in Android Studio.
To install NDK in Android Studio, navigate to configure -> Appearance & Behavior -> System Settings -> Android SDK -> SDK Tools and then select NDK.

If you need to set a specific version, download and decompress it to compile normally a GStreamer-based plugin (e.g., NNStreamer).
You can download older version from [here](https://developer.android.com/ndk/downloads/older_releases.html).

### Download GStreamer binaries

You can get the prebuilt GStreamer binaries from [here](https://gstreamer.freedesktop.org/data/pkg/android/).

For example,
```bash
$ cd $ANDROID_DEV_ROOT/gstreamer-1.0
$ curl -O https://gstreamer.freedesktop.org/data/pkg/android/1.20.0/gstreamer-1.0-android-universal-1.20.0.tar.xz
$ tar xJf gstreamer-1.0-android-universal-1.20.0.tar.xz
```

### Download NNStreamer source code and ML API source code

```bash
$ cd $ANDROID_DEV_ROOT/workspace
$ git clone https://github.com/nnstreamer/nnstreamer.git
$ git clone https://github.com/nnstreamer/api.git
```

### Build Android API

Run the build script in NNStreamer.

- Build options
  1. Target ABI: Default arm64-v8a, specify the ABI (armeabi-v7a, arm64-v8a) to be built for with `--target_abi={TARGET-ABI}`.
  2. Type: Default all.
      - `--build_type=all` to include all features.
      - `--build_type=single` to enable SingleShot API only.
      - `--build_type=lite` to get the minimized library with GStreamer core elements.
  3. Including sub-plugins: Default TensorFlow-Lite and NNFW enabled.
    To enable each neural network frameworks, you should download and set-up proper environment.
      - `--enable_tflite=yes` to build with TensorFlow-Lite.
      - `--enable_snpe=yes` to build with SNPE (Qualcomm Snapdragon Neural Processing Engine).
      - `--enable_nnfw=yes` to build with NNFW (Samsung on-device neural network inference framework).
      - `--enable_snap=yes` to build with SNAP (Samsung Neural Acceleration Platform).
  4. Enabling tracing: Default no.
      - `--enable_tracing=yes` to build with tracing and Gst-Shark.
  5. Run test: Default no. `--run_test=yes` to run the instrumentation test.
  6. Other options
      - `--nnstreamer_dir=<path>` path to NNStreamer root directory. Default `NNSTREAMER_ROOT` is used if this is not set.
      - `--ml_api_dir=<path>` path to ML API root directory. `ML_API_ROOT` is used if this is not set.
      - `--result_dir=<path>` path to build result. Default path is `ML_API_ROOT/android_lib`.
      - `--gstreamer_dir=<path>` path to GStreamer binaries. Default path is `GSTREAMER_ROOT_ANDROID`.
      - `--android_sdk_dir=<path>` path to Android SDK. Default path is `ANDROID_SDK_ROOT`.
      - `--android_ndk_dir=<path>` path to Android NDK. Default path is `ANDROID_NDK_ROOT`.

```bash
$ cd $ML_API_ROOT
$ bash ./java/build-nnstreamer-android.sh
```

After building the Android API, you can find the library(.aar) in `$ML_API_ROOT/android_lib`.
- Build result
  1. nnstreamer-[BUILD_DATE].aar: NNStreamer library for Android
  2. nnstreamer-native-[BUILD_DATE].zip: shared objects and header files for native developer

### Run the unit-test (Optional)

To run the unit-test, you will need an Android Emulator running or a physical device connected and in usb debugging mode.
Make sure to select the appropriate target ABI for the emulator.
Before running the unit-test, you should download the test model and copy it into your target device manually.

Make directory and copy test model and label files into the internal storage of your own Android target device.

You can download these files from [nnsuite testcases repository](https://github.com/nnsuite/testcases/tree/master/DeepLearningModels/).

```
# You must put the below model and label files in the internal storage of your Android target device.

## For TensorFlow Lite
# Copy {nnsuite testcases repository}/tensorflow-lite/Mobilenet_v1_1.0_224_quant/* into
{INTERNAL_STORAGE}/nnstreamer/test/mobilenet_v1_1.0_224_quant.tflite
{INTERNAL_STORAGE}/nnstreamer/test/labels.txt
{INTERNAL_STORAGE}/nnstreamer/test/orange.png
{INTERNAL_STORAGE}/nnstreamer/test/orange.raw

# Copy {nnsuite testcases repository}/tensorflow-lite/add_tflite/add.tflite into
{INTERNAL_STORAGE}/nnstreamer/test/add.tflite

## For SNPE
# Copy {nnsuite testcases repository}/snpe/inception_v3/* into
{INTERNAL_STORAGE}/nnstreamer/snpe_data/inception_v3_quantized.dlc
{INTERNAL_STORAGE}/nnstreamer/snpe_data/imagenet_slim_labels.txt
{INTERNAL_STORAGE}/nnstreamer/snpe_data/plastic_cup.jpg
{INTERNAL_STORAGE}/nnstreamer/snpe_data/plastic_cup.raw
```

To check the testcases, run the build script with an option ```--run_test=yes```.
You can find the result in ```$ML_API_ROOT/android_lib```.

```bash
$ cd $ML_API_ROOT
$ bash ./java/android/build-nnstreamer-android-lib.sh --run_test=yes
```

### Using Model File with Scoped Storage

Android keeps trying to protect app and user data on external storage. As a result, "scoped storage" is introduced in Android 10 and enhanced in Android 11.
It makes an application has [access only to the app-specific directory on external storage](https://developer.android.com/training/data-storage#scoped-storage).

With scoped storage, consider either options below:

1. (**Recommended**) Provide your model files with [`assets`](https://developer.android.com/guide/topics/resources/providing-resources#OriginalFiles).
    * Place your model files in `assets/models/`.
    * Copy it into app-specific directory using [AssetManager](https://developer.android.com/reference/android/content/res/AssetManager).
    * Use the File object with NNStreamer Java API.

    Code example:

    ```java
    /**
     * Copy files in `assets/models` into app-specific directory.
     *
     * @param context  The application context
     */
    void copyModelFromAssetsToAppDir(Context context) {
      AssetManager am = context.getResources().getAssets();
      String[] files = null;

      // Get names of files in `assets/models` directory.
      try {
        files = am.list("models");
      } catch (Exception e) {
        Log.e("TAG", "Failed to get asset file list");
        e.printStackTrace();
        return;
      }

      // Copy files into app-specific directory.
      for (String filename : files) {
        try {
          InputStream in = am.open("models/" + filename);
          String outDir = context.getFilesDir().getAbsolutePath();
          // Use `getExternalFilesDir` if you want external directory.
          File outFile = new File(outDir, filename);
          OutputStream out = new FileOutputStream(outFile);

          byte[] buffer = new byte[1024];
          int read;
          while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
          }

          in.close();
          out.flush();
          out.close();
        } catch (IOException e) {
          Log.e("TAG", "Failed to copy file: " + filename);
          e.printStackTrace();
          return;
        }
      }
    }
    ```

2. Provide your model files with absolute path.
    * Place your model files in the device's external storage.
    * (If your app targets API level 30 (Android 11) or later) Set `android:preserveLegacyExternalStorage="true"` in your `AndroidManifest.xml` to use the deprecated method `getExternalStorageDirectory`.
    * Use method `getExternalStorageDirectory` to get the File object.
    * Use the File object with NNStreamer Java API.
    * **Note:** Use this option only for test purpose. This assumes that the model files should be in the right hardcoded path in the target device.

    Code example:

    ```java
    /**
     * Return a model file in external storage.
     *
     * @return The File object
     */
    File getModelFile() {
      String root = Environment.getExternalStorageDirectory().getAbsolutePath();
      File model = new File(root + "/path/to/modelfile/sample.tflite");

      if (!model.exists()) {
        return null;
      }

      return model;
    }
    ```

### Using TensorFlow Lite NNAPI Delegate

If the TensorFlow Lite model file is provided from external storage, TensorFlow Lite NNAPI delegate fails to use available backend like GPU, DSP, or NPU.

To use NNAPI delegate properly, you should provide the model file from **internal** storage. You may copy the model files into **internal** app-specific directory (`getFilesDir`) or cache directory (`getCacheDir`).

Here are sample Java methods that copy given files into internal app-specific directory:

```java
/**
 * Copy a model file into app-specific internal storage and return it.
 *
 * @param context  The application context
 * @param model    The File object of a model file
 *
 * @return The copied File object
 */
File modelFromFilesDir(Context context, File model) {
  File appSpecificFile = new File(context.getFilesDir(), model.getName());
  appSpecificFile.mkdirs();
  // Copy the model file in external storage to app specific internal storage.
  try {
    Files.copy(model.toPath(), appSpecificFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
  } catch (Exception e) {
    e.printStackTrace();
    return null;
  }

  return appSpecificFile;
}

/**
 * Copy a model file into app-specific cache directory and return it.
 *
 * @param context  The application context
 * @param model    The File object of a model file
 *
 * @return         The copied File object
 */
File modelFromCacheDir(Context context, File model) {
  File cacheFile = new File(context.getCacheDir(), model.getName());
  cacheFile.getParentFile().mkdirs();
  // Copy the model file in external storage to app-specific cache directory.
  try {
    Files.copy(model.toPath(), cacheFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
  } catch (Exception e) {
    e.printStackTrace();
    return null;
  }

  return cacheFile;
}
```
