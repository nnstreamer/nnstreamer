if nnfw_runtime_support_is_available
  filter_sub_nnfw_sources = ['tensor_filter_nnfw.c']

  nnstreamer_filter_nnfw_deps = [nnfw_runtime_support_deps, nnstreamer_single_dep]

  # Check old function (@todo remove this definition later, nnfw ver >= 1.6.0)
  if not cc.has_header_symbol('nnfw/nnfw.h', 'nnfw_set_input_tensorinfo', dependencies: nnfw_dep)
    nnstreamer_filter_nnfw_deps += declare_dependency(compile_args: ['-DNNFW_USE_OLD_API=1'])
  endif

  nnfw_plugin_lib = shared_library('nnstreamer_filter_nnfw',
    filter_sub_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  static_library('nnstreamer_filter_nnfw',
    filter_sub_nnfw_sources,
    dependencies: nnstreamer_filter_nnfw_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  nnfw_plugin_dep = declare_dependency(link_with: nnfw_plugin_lib,
    dependencies: nnstreamer_filter_nnfw_deps)
endif

if armnn_support_is_available
  filter_sub_armnn_sources = [
    'tensor_filter_armnn.cc'
  ]

  if cxx.has_header('armnnCaffeParser/ICaffeParser.hpp')
    armnn_caffeparserlib = cxx.find_library('armnnCaffeParser', required: true)
    armnn_caffe_dep = declare_dependency(compile_args: '-DENABLE_ARMNN_CAFFE=1')
    armnn_support_deps = armnn_support_deps + [ armnn_caffe_dep, armnn_caffeparserlib ]
    message('ARMNN subplugin is built with Caffe parser.')
  endif
  if cxx.has_header('armnnTfLiteParser/ITfLiteParser.hpp')
    armnn_tflparserlib = cxx.find_library('armnnTfLiteParser', required: true)
    armnn_tfl_dep = declare_dependency(compile_args: '-DENABLE_ARMNN_TFLITE=1')
    armnn_support_deps = armnn_support_deps + [ armnn_tfl_dep, armnn_tflparserlib ]
    message('ARMNN subplugin is built with TFLite parser.')
  endif

  nnstreamer_filter_armnn_deps = [armnn_support_deps, glib_dep, nnstreamer_single_dep]

  armnn_plugin_lib = shared_library('nnstreamer_filter_armnn',
    filter_sub_armnn_sources,
    dependencies: nnstreamer_filter_armnn_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  static_library('nnstreamer_filter_armnn',
    filter_sub_armnn_sources,
    dependencies: nnstreamer_filter_armnn_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  armnn_plugin_dep = declare_dependency(link_with: armnn_plugin_lib,
    dependencies: nnstreamer_filter_armnn_deps)
endif

tf_ver_dep = disabler()
if tf_support_is_available
  tf_ver = tf_support_deps[0].version()
  tf_ver_major = tf_ver.split('.')[0]
  tf_ver_minor = tf_ver.split('.')[1]
  tf_ver_micro = tf_ver.split('.')[2]

  tf_ver_dep = declare_dependency (
    compile_args : [
      '-DTF_VERSION=' + tf_ver,
      '-DTF_VERSION_MAJOR=' + tf_ver_major,
      '-DTF_VERSION_MINOR=' + tf_ver_minor,
      '-DTF_VERSION_MICRO=' + tf_ver_micro,
    ],
  )
  message('TensorFlow version: ' + tf_ver)

  filter_sub_tf_sources = ['tensor_filter_tensorflow.cc']

  nnstreamer_filter_tf_deps = [tf_support_deps, nnstreamer_single_dep]
  nnstreamer_filter_tf_deps += tf_ver_dep

  shared_library('nnstreamer_filter_tensorflow',
    filter_sub_tf_sources,
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow',
    filter_sub_tf_sources,
    dependencies: nnstreamer_filter_tf_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

tflite_ver_dep = disabler()
if tflite_support_is_available
  tflite_ver = tflite_support_deps[0].version()
  tflite_ver_major = tflite_ver.split('.')[0]
  tflite_ver_minor = tflite_ver.split('.')[1]
  tflite_ver_micro = tflite_ver.split('.')[2]

  tflite_ver_dep = declare_dependency (
    compile_args : [
      '-DTFLITE_VERSION=' + tflite_ver,
      '-DTFLITE_VERSION_MAJOR=' + tflite_ver_major,
      '-DTFLITE_VERSION_MINOR=' + tflite_ver_minor,
      '-DTFLITE_VERSION_MICRO=' + tflite_ver_micro,
    ],
  )
  message('TensorFlow-lite version: ' + tflite_ver)

  if not get_option('skip-tflite-flatbuf-check') and not flatbuf_support_is_available
    flatbuf_dep = dependency('flatbuffers', required : true, \
        not_found_message : 'flatbuf devel package should be install to build the tensorflow lite subplugin.')
  endif

  filter_sub_tflite_sources = ['tensor_filter_tensorflow_lite.cc']

  nnstreamer_filter_tflite_deps = [tflite_support_deps, thread_dep, libdl_dep, glib_dep, nnstreamer_single_dep]

  # Since tf-1.13, tflite has moved from contrib to core
  tflite_h_prefix='#include <tensorflow/lite/model.h>'
  if tflite_ver_minor.to_int() < 13
    tflite_h_prefix='#include <tensorflow/contrib/lite/model.h>'
  endif

  tflite_compile_args = ['-DTFLITE_SUBPLUGIN_NAME="tensorflow1-lite"']
  if cxx.has_type('kTfLiteInt8', prefix : tflite_h_prefix)
    tflite_compile_args += '-DTFLITE_INT8=1'
  endif
  if cxx.has_type('kTfLiteInt16', prefix : tflite_h_prefix)
    tflite_compile_args += '-DTFLITE_INT16=1'
  endif
  if cxx.has_type('kTfLiteFloat16', prefix : tflite_h_prefix)
    tflite_compile_args += '-DTFLITE_FLOAT16=1'
  endif
  if cxx.has_type('kTfLiteComplex64', prefix : tflite_h_prefix)
    tflite_compile_args += '-DTFLITE_COMPLEX64=1'
  endif

  tflite_extra_dep = declare_dependency(
    compile_args : tflite_compile_args
  )

  nnstreamer_filter_tflite_deps += tflite_extra_dep
  nnstreamer_filter_tflite_deps += tflite_ver_dep

  shared_library('nnstreamer_filter_tensorflow1-lite',
    filter_sub_tflite_sources,
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow1-lite',
    filter_sub_tflite_sources,
    dependencies: nnstreamer_filter_tflite_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

tflite2_ver_dep = disabler()
if tflite2_support_is_available
  tflite2_ver = tflite2_support_deps[0].version()
  tflite2_ver_major = tflite2_ver.split('.')[0]
  tflite2_ver_minor = tflite2_ver.split('.')[1]
  tflite2_ver_micro = tflite2_ver.split('.')[2]

  tflite2_ver_dep = declare_dependency (
    compile_args : [
      '-DTFLITE_VERSION=' + tflite2_ver,
      '-DTFLITE_VERSION_MAJOR=' + tflite2_ver_major,
      '-DTFLITE_VERSION_MINOR=' + tflite2_ver_minor,
      '-DTFLITE_VERSION_MICRO=' + tflite2_ver_micro,
    ],
  )
  message('TensorFlow2-lite version: ' + tflite2_ver)

  # check flatbuffers header
  ## Note that tizen tflite package has its own flatbuffers header internally.
  if build_platform != 'tizen' and not cxx.check_header('flatbuffers/flatbuffers.h')
    error('flatbuffers header files are required to build tensorflow2-lite subplugin. Please install compatible version for tensorflow-lite library. Or disable tflite2-support in meson_options.txt')
  endif

  filter_sub_tflite2_sources = ['tensor_filter_tensorflow_lite.cc']

  nnstreamer_filter_tflite2_deps = [tflite2_support_deps, thread_dep, libdl_dep, nnstreamer_single_dep]

  tflite2_compile_args = []
  tflite2_compile_args += '-DTFLITE_INT8=1'
  tflite2_compile_args += '-DTFLITE_INT16=1'
  tflite2_compile_args += '-DTFLITE_FLOAT16=1'
  tflite2_compile_args += '-DTFLITE_COMPLEX64=1'

  if cc.has_header('tensorflow2/lite/model.h')
    tflite2_compile_args += '-DUSE_TENSORFLOW2_HEADER_PATH=1'
  endif

  # Check each delegate linking with given tflite library
  ## xnnpack delegate
  if cxx.links('''
        #include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
        TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault ();
        int main() {return 0;}
      ''', dependencies : [tflite2_support_deps, thread_dep], name : 'xnnpack delegate')
    tflite2_compile_args += '-DTFLITE_XNNPACK_DELEGATE_SUPPORTED'
  endif

  ## gpu delegate
  if cxx.links('''
        #include "tensorflow/lite/delegates/gpu/delegate.h"
        TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default ();
        int main() {return 0;}
      ''', dependencies : [tflite2_support_deps, libdl_dep], name : 'gpu delegate')
    tflite2_compile_args += '-DTFLITE_GPU_DELEGATE_SUPPORTED'
  endif

  ## nnapi delegate
  if cxx.links('''
        #include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
        int main() {
          tflite::StatefulNnApiDelegate *nnapi_delegate = new tflite::StatefulNnApiDelegate ();
          delete nnapi_delegate;
          return 0;
        }
      ''', dependencies : [tflite2_support_deps], name : 'nnapi delegate')
    tflite2_compile_args += '-DTFLITE_NNAPI_DELEGATE_SUPPORTED'
  endif

  ## external delegate
  if cxx.links('''
      #include "tensorflow/lite/delegates/external/external_delegate.h"
      TfLiteExternalDelegateOptions external_options = TfLiteExternalDelegateOptionsDefault ("");
      int main() {return 0;}
    ''', dependencies : [tflite2_support_deps, libdl_dep], name : 'external delegate')
    tflite2_compile_args += '-DTFLITE_EXTERNAL_DELEGATE_SUPPORTED'
  endif

  tflite2_custom_compile_args = tflite2_compile_args
  tflite2_compile_args += '-DTFLITE_SUBPLUGIN_NAME="tensorflow2-lite"'

  tflite2_extra_dep = declare_dependency(
    compile_args : tflite2_compile_args
  )

  nnstreamer_filter_tflite2_deps += tflite2_extra_dep
  nnstreamer_filter_tflite2_deps += tflite2_ver_dep

  shared_library('nnstreamer_filter_tensorflow2-lite',
    filter_sub_tflite2_sources,
    dependencies: nnstreamer_filter_tflite2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorflow2-lite',
    filter_sub_tflite2_sources,
    dependencies: nnstreamer_filter_tflite2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )


  if (get_option('tflite2-custom-support').enabled() or get_option('tflite2-custom-support').auto()) and not meson.is_cross_build()
    ## Create libtensorflow2-lite-custom.so support.

    ## @todo This assumes the same tflite version with the above.
    ##       If the user has downloaded libtensorflow-lite manually, it may incur inconsistency.
    ## @todo This does not use pkgconfig or find_library/meson-dep info
    ##       because it does not allow querying libpath and static/shared of a dependency object.
    ## @todo If we are allowed to use meson > 0.53, use "FS" module!

    filepaths = [
        join_paths(meson.current_source_dir(), 'libtensorflow2-lite.so'),
        join_paths(meson.current_build_dir(), 'libtensorflow2-lite.so'),
        join_paths(nnstreamer_libdir, 'libtensorflow2-lite.so'),
        '/usr/lib/libtensorflow2-lite.so',
        '/usr/lib64/libtensorflow2-lite.so',
        './libtensorflow2-lite.so'
    ]
    filepath = ''
    foreach file : filepaths
      if run_command('bash', '-c', '[ -f "' + file + '" ]', check : false).returncode() == 0
        message('Found: ' + file)
        filepath = file
        configure_file (
          input: file,
          output: 'libtensorflow2-lite-custom.so',
          copy: true
        )
        break
      endif
    endforeach

    if filepath == ''
      ## No shared tfl lib found. Get static lib and convert it
      custom_tflite_lib = join_paths(meson.current_build_dir(), 'libtensorflow2-lite-custom.so')

      filepaths = [join_paths(nnstreamer_libdir, 'libtensorflow2-lite.a'),
          '/usr/lib/libtensorflow2-lite.a',
          '/usr/lib64/libtensorflow2-lite.a',
          './libtensorflow2-lite.a',
          join_paths(meson.current_build_dir(), 'libtensorflow2-lite.a'),
          join_paths(meson.current_source_dir(), 'libtensorflow2-lite.a')
      ]

      foreach file : filepaths
        if run_command('bash', '-c', '[ -f "'+ file + '" ]', check : false).returncode() == 0
          somake = run_command('gcc', '-shared', '-o', custom_tflite_lib, '-Wl,--whole-archive', file, '-Wl,--no-whole-archive', check : false)
          if somake.returncode() == 0
            message('Successfully created libtensorflow2-lite-custom.so from archive: ' + file)
            filepath = file
            break
          else
            error('Cannot create .so from: ' + file)
          endif
        endif
      endforeach
    endif

    if filepath != ''
      tflite2_custom_compile_args += '-DTFLITE_SUBPLUGIN_NAME="tensorflow2-lite-custom"'
      tflite2_custom_extra_dep = declare_dependency( compile_args : tflite2_custom_compile_args )
      tflc_dep = declare_dependency (
        link_args: ['-L' + meson.current_build_dir(), '-ltensorflow2-lite-custom']
      )
      shared_library('nnstreamer_filter_tensorflow2-lite-custom',
        filter_sub_tflite2_sources,
        dependencies: [tflc_dep, thread_dep, libdl_dep, nnstreamer_single_dep, tflite2_ver_dep, tflite2_custom_extra_dep],
        install: true,
        install_dir:filter_subplugin_install_dir
      )
    else # no custom mode built
      if get_option('tflite2-custom-support').enabled()
        error('Cannot create tflite2-custom tensor-filter subplugin. No valid .so/.a of tensorflow2-lite found.')
      endif
    endif
  endif
else # no tflite2
  if get_option('tflite2-custom-support').enabled()
    error('tflite2-custom-support requires tflite2-support')
  endif
endif

if pytorch_support_is_available
  if cxx.get_id() != 'gcc'
    error('Pytorch headers (Array.h and reverse_iterator.h) require GCC for license issues. Use GCC or disable pytorch with -Dpytorch-support=disabled.')
  endif

  filter_sub_torch_sources = ['tensor_filter_pytorch.cc']

  nnstreamer_filter_torch_deps = [pytorch_support_deps, nnstreamer_single_dep]
  if cxx.has_argument('-Wno-unused-parameter')
    # pytorch sources contain unused parameters
    nnstreamer_filter_torch_deps += declare_dependency(compile_args: ['-Wno-unused-parameter'])
  endif

  shared_library('nnstreamer_filter_pytorch',
    filter_sub_torch_sources,
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_pytorch',
    filter_sub_torch_sources,
    dependencies: nnstreamer_filter_torch_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if caffe2_support_is_available
  filter_sub_caffe2_sources = ['tensor_filter_caffe2.cc']

  nnstreamer_filter_caffe2_deps = [caffe2_support_deps, protobuf_dep, glib_dep, nnstreamer_single_dep]

  shared_library('nnstreamer_filter_caffe2',
    filter_sub_caffe2_sources,
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_caffe2',
    filter_sub_caffe2_sources,
    dependencies: nnstreamer_filter_caffe2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if deepview_rt_support_is_available
  nnstreamer_filter_deepview_rt_deps = [deepview_rt_support_deps, glib_dep, nnstreamer_single_dep]

  filter_sub_deepview_rt_sources = ['tensor_filter_deepview_rt.cc']

  shared_library('nnstreamer_filter_deepview-rt',
    filter_sub_deepview_rt_sources,
    dependencies: nnstreamer_filter_deepview_rt_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_deepview-rt',
    filter_sub_deepview_rt_sources,
    dependencies: nnstreamer_filter_deepview_rt_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if have_python3
  filter_sub_python3_sources = ['tensor_filter_python3.cc']

  shared_library('nnstreamer_filter_python3',
    filter_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_python3',
    filter_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if mvncsdk2_support_is_available
  filter_sub_mvncsdk2_sources = [
    'tensor_filter_movidius_ncsdk2.c'
  ]

  nnstreamer_filter_mvncsdk2_deps = [mvncsdk2_support_deps, glib_dep, nnstreamer_single_dep]

  shared_library('nnstreamer_filter_movidius-ncsdk2',
    filter_sub_mvncsdk2_sources,
    dependencies: nnstreamer_filter_mvncsdk2_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_movidius-ncsdk2',
    filter_sub_mvncsdk2_sources,
    dependencies: nnstreamer_filter_mvncsdk2_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if get_option('enable-cppfilter')
  filter_sub_cpp_sources = ['tensor_filter_cpp.cc']

  nnstreamer_cpp_lib = shared_library('nnstreamer_filter_cpp',
    filter_sub_cpp_sources,
    dependencies: nnstreamer_single_dep,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
  install_headers(['tensor_filter_cpp.hh'],
    subdir: 'nnstreamer'
  )
  configure_file(input: 'nnstreamer-cpp.pc.in', output: 'nnstreamer-cpp.pc',
    install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
    configuration: nnstreamer_install_conf
  )
  nnstreamer_cpp_dep = declare_dependency(link_with: nnstreamer_cpp_lib,
    dependencies: nnstreamer_single_dep,
    include_directories: include_directories('.')
  )
endif

if get_option('enable-edgetpu')
  if not get_option('enable-filter-cpp-class')
    error ('enable-filter-cpp-class should be set as \'true\' to build the tensor filter for Edge TPU.')
  endif

  edgetpu_dep = dependency('edgetpu', required: false)
  if not edgetpu_dep.found()
    # Since the development package for Ubuntu does not have pkgconfig file,
    # check that the required header and library files exist in the system
    # include and lib directories.
    if cxx.has_header('edgetpu.h')
      edgetpu_dep = declare_dependency (
        dependencies: cxx.find_library('edgetpu'),
      )
    else
      error('failed to resolve the build dependency of the tensor filter for Edge TPU.')
    endif
  endif

  nnstreamer_filter_edgetpu_deps = [glib_dep, nnstreamer_single_dep, edgetpu_dep, libdl_dep]

  if tflite2_support_is_available
    nnstreamer_filter_edgetpu_deps += [tflite2_support_deps, tflite2_ver_dep]
    message('build Edge TPU sub-plugin with tensorflow2-lite')
  elif tflite_support_is_available
    nnstreamer_filter_edgetpu_deps += [tflite_support_deps, tflite_ver_dep]
    message('build Edge TPU sub-plugin with tensorflow1-lite')
  else
    error ('tflite2-support or tflite-support should be set as \'enabled\' to build the tensor filter for Edge TPU.')
  endif

  filter_sub_edgetpu_sources = ['tensor_filter_edgetpu.cc']

  rpath_edgetpu_test_helper = ''
  if get_option('enable-test')
    '''
    In order to use the test helper library, add build_rpath.
    This temporarily overrides the runtime dependency on edge tpu and
    it is removed after installation.
    '''
    rpath_edgetpu_test_helper = join_paths(meson.build_root(), 'tests/nnstreamer_filter_edgetpu')
  endif

  shared_library('nnstreamer_filter_edgetpu',
    filter_sub_edgetpu_sources,
    dependencies: nnstreamer_filter_edgetpu_deps,
    install: true,
    install_dir: filter_subplugin_install_dir,
    build_rpath: rpath_edgetpu_test_helper
  )
endif

if get_option('enable-openvino')
  openvino_deps = []
  openvino_cpp_args = []
  openvino_deps += dependency('openvino', required: true)
  if cxx.has_header('ext_list.hpp')
    openvino_cpuext_lib = cxx.find_library('cpu_extension', required: true)
    if openvino_cpuext_lib.found()
      openvino_cpp_args += '-D__OPENVINO_CPU_EXT__=1'
      openvino_deps += openvino_cpuext_lib
    else
      error('Cannot find the shared object, libcpu_extension.')
    endif
  endif
  filter_sub_openvino_sources = ['tensor_filter_openvino.cc']

  nnstreamer_filter_openvino_deps = [glib_dep, nnstreamer_single_dep, openvino_deps]

  nnstreamer_filter_openvino_so_lib = shared_library('nnstreamer_filter_openvino',
    filter_sub_openvino_sources,
    cpp_args: openvino_cpp_args,
    dependencies: nnstreamer_filter_openvino_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_openvino',
    filter_sub_openvino_sources,
    cpp_args: openvino_cpp_args,
    dependencies: nnstreamer_filter_openvino_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  nnstreamer_filter_openvino_dep = declare_dependency(
    link_with: nnstreamer_filter_openvino_so_lib,
    include_directories: include_directories('.')
  )
endif

if get_option('enable-vivante')
  subdir('vivante')
endif

if get_option('enable-mediapipe')
  if host_machine.system() == 'linux' and host_machine.cpu_family() == 'x86_64'
    opencv_path = 'bazel-bin/_solib_k8/_U@linux_Uopencv_S_S_Copencv___Uexternal_Slinux_Uopencv_Slib_Sx86_U64-linux-gnu'
  else
    error('Not Supported System & Architecture. Linux x86_64 is required')
  endif

  cmd = run_command('sh', '-c', 'echo $MEDIAPIPE_HOME', check : true)
  MEDIAPIPE_HOME = cmd.stdout().strip()
  if MEDIAPIPE_HOME == ''
    error('MEDIAPIPE_HOME is not set')
  endif
  message('MEDIAPIPE_HOME: ' + MEDIAPIPE_HOME)
  # TODO: This is not a portable way to check if a directory exists
  # After bumping up to 0.53.0, the following line can be replaced with one that uses FS module
  mediapipe_dir_exist = run_command('[', '-d', MEDIAPIPE_HOME, ']', check : false).returncode()
  if mediapipe_dir_exist != 0
    error('Cannot find MEDIAPIPE_HOME: ' + MEDIAPIPE_HOME)
  endif

  mediapipe_incdir = include_directories(
    MEDIAPIPE_HOME,
    join_paths(MEDIAPIPE_HOME, 'bazel-bin'),
    join_paths(MEDIAPIPE_HOME, 'bazel-mediapipe/external/eigen_archive'),
    join_paths(MEDIAPIPE_HOME, 'bazel-mediapipe/external/com_google_absl'),
  )

  run_command('sh', '-c', meson.source_root() + '/tools/development/gen_mediapipe_libs.sh ' + nnstreamer_libdir, check : true)

  mediapipe_internal_dep = cxx.find_library('mediapipe_internal', dirs: nnstreamer_libdir)
  mediapipe_external_dep = cxx.find_library('mediapipe_external', dirs: nnstreamer_libdir)

  opencv_core_dep = cc.find_library('opencv_core', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_video_dep = cc.find_library('opencv_video', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_imgcodecs_dep = cc.find_library('opencv_imgcodecs', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_imgproc_dep = cc.find_library('opencv_imgproc', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_calib3d_dep = cc.find_library('opencv_calib3d', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_highgui_dep = cc.find_library('opencv_highgui', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_videoio_dep = cc.find_library('opencv_videoio', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))
  opencv_features2d_dep = cc.find_library('opencv_features2d', dirs: join_paths(MEDIAPIPE_HOME, opencv_path))

  nnstreamer_filter_mediapipe_deps = [
    glib_dep,
    nnstreamer_single_dep,
    opencv_core_dep,
    opencv_video_dep,
    opencv_imgcodecs_dep,
    opencv_imgproc_dep,
    opencv_calib3d_dep,
    opencv_highgui_dep,
    opencv_videoio_dep,
    opencv_features2d_dep,
    mediapipe_external_dep,
    mediapipe_internal_dep
  ]

  filter_sub_mp_sources = ['tensor_filter_mediapipe.cc']

  shared_library('nnstreamer_filter_mediapipe',
    filter_sub_mp_sources,
    dependencies: nnstreamer_filter_mediapipe_deps,
    include_directories: mediapipe_incdir,
    install: true,
    install_dir: filter_subplugin_install_dir
  )
endif

if snpe_support_is_available
  # Decide source code file wrt snpe api version.
  if snpe_api_version == 2
    filter_sub_snpe_sources = ['tensor_filter_snpe.cc']
  else # snpe_api_version == 1
    filter_sub_snpe_sources = ['tensor_filter_snpe_v1.cc']
  endif

  nnstreamer_filter_snpe_deps = [glib_dep, nnstreamer_single_dep, snpe_support_deps]
  if cxx.has_argument('-Wno-deprecated-declarations')
    nnstreamer_filter_snpe_deps += declare_dependency(compile_args: ['-Wno-deprecated-declarations'])
  endif

  shared_library('nnstreamer_filter_snpe',
    filter_sub_snpe_sources,
    dependencies: nnstreamer_filter_snpe_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_snpe',
    filter_sub_snpe_sources,
    dependencies: nnstreamer_filter_snpe_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if qnn_support_is_available
  filter_sub_qnn_sources = ['tensor_filter_qnn.cc']

  nnstreamer_filter_qnn_deps = [glib_dep, nnstreamer_single_dep, qnn_support_deps]
  if cxx.has_argument('-Wno-missing-braces')
    nnstreamer_filter_qnn_deps += declare_dependency(compile_args: ['-Wno-missing-braces'])
  endif

  shared_library('nnstreamer_filter_qnn',
    filter_sub_qnn_sources,
    dependencies: nnstreamer_filter_qnn_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_qnn',
    filter_sub_qnn_sources,
    dependencies: nnstreamer_filter_qnn_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if tensorrt_support_is_available
  filter_sub_tensorrt_sources = ['tensor_filter_tensorrt.cc']

  nnstreamer_filter_tensorrt_deps = [glib_dep, nnstreamer_single_dep, tensorrt_support_deps]

  shared_library('nnstreamer_filter_tensorrt',
    filter_sub_tensorrt_sources,
    dependencies: nnstreamer_filter_tensorrt_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorrt',
    filter_sub_tensorrt_sources,
    dependencies: nnstreamer_filter_tensorrt_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if tensorrt10_support_is_available
  filter_sub_tensorrt10_sources = ['tensor_filter_tensorrt10.cc']

  nnstreamer_filter_tensorrt10_deps = [glib_dep, nnstreamer_single_dep, tensorrt10_support_deps]

  shared_library('nnstreamer_filter_tensorrt10',
    filter_sub_tensorrt10_sources,
    dependencies: nnstreamer_filter_tensorrt10_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tensorrt10',
    filter_sub_tensorrt10_sources,
    dependencies: nnstreamer_filter_tensorrt10_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if lua_support_is_available
  if lua_support_deps[0].version().version_compare('>=5.3')
    message ('tensor-filter::lua does not support Lua >= 5.3, yet. Fix #3531 first.')
    lua_support_is_available = disabler()
  endif
endif

if lua_support_is_available
  filter_sub_lua_sources = ['tensor_filter_lua.cc']

  nnstreamer_filter_lua_deps = [glib_dep, nnstreamer_single_dep, lua_support_deps]

  shared_library('nnstreamer_filter_lua',
    filter_sub_lua_sources,
    dependencies: nnstreamer_filter_lua_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_lua',
    filter_sub_lua_sources,
    dependencies: nnstreamer_filter_lua_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if tvm_support_is_available
  nnstreamer_filter_tvm_deps = [tvm_support_deps, glib_dep, nnstreamer_dep]

  filter_sub_tvm_sources = ['tensor_filter_tvm.cc']

  shared_library('nnstreamer_filter_tvm',
    filter_sub_tvm_sources,
    dependencies: nnstreamer_filter_tvm_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_tvm',
    filter_sub_tvm_sources,
    dependencies: nnstreamer_filter_tvm_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if trix_engine_support_is_available
  nnstreamer_filter_trix_engine_deps = [trix_engine_support_deps, glib_dep, nnstreamer_single_dep]

  filter_sub_trix_engine_sources = ['tensor_filter_trix_engine.cc']

  shared_library('nnstreamer_filter_trix-engine',
    filter_sub_trix_engine_sources,
    dependencies: nnstreamer_filter_trix_engine_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_trix-engine',
    filter_sub_trix_engine_sources,
    dependencies: nnstreamer_filter_trix_engine_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if mxnet_support_is_available
  if not get_option('enable-filter-cpp-class')
    error ('enable-filter-cpp-class should be set as \'true\' to build the tensor filter for MXNet.')
  endif

  filter_sub_mxnet_sources = [
    'tensor_filter_mxnet.cc'
  ]

  nnstreamer_filter_mxnet_deps = [mxnet_support_deps, glib_dep, nnstreamer_single_dep, libdl_dep]
  if cxx.has_argument('-Wno-non-virtual-dtor')
    nnstreamer_filter_mxnet_deps += declare_dependency(compile_args: ['-Wno-non-virtual-dtor'])
  endif

  nnstreamer_filter_mxnet_so_lib = shared_library('nnstreamer_filter_mxnet',
    filter_sub_mxnet_sources,
    dependencies: nnstreamer_filter_mxnet_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  nnstreamer_filter_mxnet_dep = declare_dependency(
    link_with: nnstreamer_filter_mxnet_so_lib,
    include_directories: include_directories('.')
  )
endif

if ncnn_support_is_available
  filter_sub_ncnn_sources = ['tensor_filter_ncnn.cc']

  nnstreamer_filter_ncnn_deps = [ncnn_support_deps, glib_dep, nnstreamer_single_dep]

  shared_library('nnstreamer_filter_ncnn',
    filter_sub_ncnn_sources,
    dependencies: nnstreamer_filter_ncnn_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_ncnn',
    filter_sub_ncnn_sources,
    dependencies: nnstreamer_filter_ncnn_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if executorch_support_is_available
  filter_sub_executorch_sources = ['tensor_filter_executorch.cc']

  nnstreamer_filter_executorch_deps = [executorch_support_deps, glib_dep, nnstreamer_single_dep]

  shared_library('nnstreamer_filter_executorch',
    filter_sub_executorch_sources,
    dependencies: nnstreamer_filter_executorch_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_executorch',
    filter_sub_executorch_sources,
    dependencies: nnstreamer_filter_executorch_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if executorch_llama_support_is_available
  filter_sub_executorch_llama_sources = ['tensor_filter_executorch_llama.cc']
  nnstreamer_filter_executorch_llama_deps = [executorch_llama_support_deps, glib_dep, nnstreamer_single_dep]

  executorch_llama_compile_args = cxx.get_supported_arguments([
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations'
  ])

  nnstreamer_filter_executorch_llama_deps += declare_dependency(compile_args: executorch_llama_compile_args)

  shared_library('nnstreamer_filter_executorch-llama',
    filter_sub_executorch_llama_sources,
    dependencies: nnstreamer_filter_executorch_llama_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_executorch-llama',
    filter_sub_executorch_llama_sources,
    dependencies: nnstreamer_filter_executorch_llama_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if onnxruntime_support_is_available
  filter_sub_onnxruntime_sources = ['tensor_filter_onnxruntime.cc']

  nnstreamer_filter_onnxruntime_deps = [glib_dep, nnstreamer_single_dep, onnxruntime_support_deps]

  shared_library('nnstreamer_filter_onnxruntime',
    filter_sub_onnxruntime_sources,
    dependencies: nnstreamer_filter_onnxruntime_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_onnxruntime',
    filter_sub_onnxruntime_sources,
    dependencies: nnstreamer_filter_onnxruntime_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if dali_support_is_available
  filter_sub_dali_sources = ['tensor_filter_dali.cc']

  nnstreamer_filter_dali_deps = [glib_dep, nnstreamer_single_dep, cuda_dep, cudart_dep, dali_support_deps]

  shared_library('nnstreamer_filter_dali',
    filter_sub_dali_sources,
    dependencies: nnstreamer_filter_dali_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_dali',
    filter_sub_dali_sources,
    dependencies: nnstreamer_filter_dali_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if llamacpp_support_is_available
  filter_sub_llamacpp_sources = ['tensor_filter_llamacpp.cc']
  libggml_dep = cc.find_library('ggml')
  nnstreamer_filter_llamacpp_deps = [glib_dep, nnstreamer_single_dep, llamacpp_support_deps, libggml_dep]

  shared_library('nnstreamer_filter_llamacpp',
    filter_sub_llamacpp_sources,
    dependencies: nnstreamer_filter_llamacpp_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_llamacpp',
    filter_sub_llamacpp_sources,
    dependencies: nnstreamer_filter_llamacpp_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if llama2c_support_is_available
  filter_sub_llama2c_sources = ['tensor_filter_llama2.cc']

  nnstreamer_filter_llama2c_deps = [glib_dep, nnstreamer_single_dep, llama2c_support_deps]

  shared_library('nnstreamer_filter_llama2c',
    filter_sub_llama2c_sources,
    dependencies: nnstreamer_filter_llama2c_deps,
    install: true,
    install_dir: filter_subplugin_install_dir
  )

  static_library('nnstreamer_filter_llama2c',
    filter_sub_llama2c_sources,
    dependencies: nnstreamer_filter_llama2c_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
