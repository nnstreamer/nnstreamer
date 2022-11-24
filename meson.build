# If you are using Ubuntu/Xenial, Do "force-version" on meson to get the required version.
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
  version: '2.3.0',
  license: ['LGPL-2.1'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=true',
    'warning_level=2',
    'c_std=gnu89',
    'cpp_std=c++14'
  ]
)

add_project_arguments('-DVERSION="' + meson.project_version() + '"', language: ['c', 'cpp'])
version_split = meson.project_version().split('.')
add_project_arguments('-DVERSION_MAJOR="' + version_split[0] + '"', language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MINOR="' + version_split[1] + '"', language: ['c', 'cpp'])
add_project_arguments('-DVERSION_MICRO="' + version_split[2] + '"', language: ['c', 'cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
build_platform = ''
so_ext = 'so'

if get_option('enable-tizen')
  # Pass __TIZEN__ to the compiler
  add_project_arguments('-D__TIZEN__=1', language: ['c', 'cpp'])
  build_platform = 'tizen'

  tizenVmajor = get_option('tizen-version-major')
  add_project_arguments('-DTIZENVERSION='+tizenVmajor.to_string(), language: ['c', 'cpp'])
  dlog_dep = dependency('dlog')
elif not meson.is_cross_build()
  if cc.get_id() == 'clang' and cxx.get_id() == 'clang'
    if build_machine.system() == 'darwin'
      # Pass __MACOS__ to the compiler
      add_project_arguments('-D__MACOS__=1', language: ['c', 'cpp'])
      build_platform = 'macos'
      so_ext = 'dylib'
    endif
  endif
endif

# Define warning flags for c and cpp
warning_flags = [
  '-Wmissing-braces',
  '-Wmaybe-uninitialized',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Winit-self',
  '-Waddress',
  '-Wno-multichar',
  '-Wvla',
  '-Wpointer-arith'
]

warning_c_flags = [
  '-Wmissing-declarations',
  '-Wmissing-include-dirs',
  '-Wmissing-prototypes',
  '-Wnested-externs',
  '-Waggregate-return',
  '-Wold-style-definition',
  '-Wdeclaration-after-statement'
]

# Setup warning flags for c and cpp
foreach extra_arg : warning_flags
  if cc.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'c')
  endif
  if cxx.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'cpp')
  endif
endforeach

foreach extra_arg : warning_c_flags
  if cc.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'c')
  endif
endforeach

gst_api_verision = '1.0'

# Set install path
nnstreamer_prefix = get_option('prefix')
nnstreamer_libdir = join_paths(nnstreamer_prefix, get_option('libdir'))
nnstreamer_bindir = join_paths(nnstreamer_prefix, get_option('bindir'))
nnstreamer_includedir = join_paths(nnstreamer_prefix, get_option('includedir'))
nnstreamer_inidir = join_paths(nnstreamer_prefix, get_option('sysconfdir'))
# join_paths drops first arg if second arg is absolute path.

# nnstreamer plugins path
plugins_install_dir = join_paths(nnstreamer_libdir, 'gstreamer-' + gst_api_verision)

# nnstreamer sub-plugins path
if get_option('subplugindir') == ''
  subplugin_install_prefix = join_paths(nnstreamer_prefix, get_option('libdir'), 'nnstreamer')
else
  subplugin_install_prefix = get_option('subplugindir')
endif
filter_subplugin_install_dir = join_paths(subplugin_install_prefix, 'filters')
decoder_subplugin_install_dir = join_paths(subplugin_install_prefix, 'decoders')
customfilter_install_dir = join_paths(subplugin_install_prefix, 'customfilters')
converter_subplugin_install_dir = join_paths(subplugin_install_prefix, 'converters')
unittest_base_dir = join_paths(nnstreamer_bindir, 'unittest-nnstreamer')

# Set default configuration
nnstreamer_conf = configuration_data()
nnstreamer_conf.set('VERSION', meson.project_version())
nnstreamer_conf.set('PREFIX', nnstreamer_prefix)
nnstreamer_conf.set('EXEC_PREFIX', nnstreamer_bindir)
nnstreamer_conf.set('LIB_INSTALL_DIR', nnstreamer_libdir)
nnstreamer_conf.set('GST_INSTALL_DIR', plugins_install_dir)
nnstreamer_conf.set('INCLUDE_INSTALL_DIR', nnstreamer_includedir)
nnstreamer_conf.set('SUBPLUGIN_INSTALL_PREFIX', subplugin_install_prefix)

# Set framework priority about model file extension when automatically selecting framework for tensor filter.
nnstreamer_conf.set('FRAMEWORK_PRIORITY_TFLITE', get_option('framework-priority-tflite'))
nnstreamer_conf.set('FRAMEWORK_PRIORITY_NB', get_option('framework-priority-nb'))
nnstreamer_conf.set('FRAMEWORK_PRIORITY_BIN', get_option('framework-priority-bin'))

# Set the alias for backward compatibility
nnstreamer_conf.set('TRIX_ENGINE_ALIAS', get_option('trix-engine-alias'))

# Define default conf file
add_project_arguments('-DNNSTREAMER_CONF_FILE="' + join_paths(nnstreamer_inidir, 'nnstreamer.ini') + '"', language: 'c')

# Dependencies
glib_dep = dependency('glib-2.0')
if glib_dep.version().version_compare('>= 2.68.0')
  add_project_arguments('-DGLIB_USE_G_MEMDUP2', language: ['c', 'cpp'])
endif
gobject_dep = dependency('gobject-2.0')
gmodule_dep = dependency('gmodule-2.0')
gio_dep = dependency('gio-2.0')
gst_dep = dependency('gstreamer-' + gst_api_verision)
gst_base_dep = dependency('gstreamer-base-' + gst_api_verision)
gst_controller_dep = dependency('gstreamer-controller-' + gst_api_verision)
gst_video_dep = dependency('gstreamer-video-' + gst_api_verision)
gst_audio_dep = dependency('gstreamer-audio-' + gst_api_verision)
gst_app_dep = dependency('gstreamer-app-' + gst_api_verision)
gst_check_dep = dependency('gstreamer-check-' + gst_api_verision)

libm_dep = cc.find_library('m') # cmath library
libdl_dep = cc.find_library('dl') # DL library
thread_dep = dependency('threads') # pthread for tensorflow-lite

# Protobuf
protobuf_dep = dependency('protobuf', version: '>= 3.6.1', required: false)

# Flatbuffers compiler and libraries
flatc = find_program('flatc', required : get_option('flatbuf-support'))
flatbuf_dep = disabler()
flatbuf_version_check_dep = disabler()
if flatc.found()
  # TODO: After bumping up meson version to 0.62.0, we can use flatc.version()
  # Please refer https://mesonbuild.com/Reference-manual_returned_external_program.html#external_programversion
  flatc_ver = run_command(flatc, '--version', check : true).stdout().split()[2]
  flatbuf_dep = dependency('flatbuffers', version: flatc_ver,
      required : get_option('flatbuf-support'))
  flatbuf_version_check_dep = dependency('flatbuffers', version: '>=2.0.0',
      required : get_option('flatbuf-support'))
endif

# Protobuf compiler
pb_comp = find_program('protoc', required: get_option('protobuf-support'))

#orc
pg_orcc = find_program('orcc', required: get_option('orcc-support'))
orc_dep = dependency('orc-0.4', version: '>= 0.4.17', required: get_option('orcc-support'))

## nnfw
nnfw_dep = dependency('', required: false)
if not get_option('nnfw-runtime-support').disabled()
  nnfw_dep = dependency('nnfw', required: false)
  if not nnfw_dep.found()
    nnfw_dep = cc.find_library('nnfw-dev', required: get_option('nnfw-runtime-support'))
  endif
endif

# snpe
snpe_dep = dependency('', required: false)
if not get_option('snpe-support').disabled()
  # Check whether the platform supports snpe
  snpe_dep = dependency('snpe', required: false)
  if (not snpe_dep.found())
    # TODO: support various arch.
    if host_machine.system() != 'linux' or host_machine.cpu_family() != 'x86_64'
      message('Not Supported System & Architecture. Linux x86_64 is required for snpe sub-plugin')
    else
      # Check whether $SNPE_ROOT (where SNPE SDK is located) is set.
      cmd = run_command('sh', '-c', 'echo $SNPE_ROOT', check : true)
      SNPE_ROOT = cmd.stdout().strip()
      if SNPE_ROOT != ''
        message('Got $SNPE_ROOT: @0@'.format(SNPE_ROOT))
        # TODO: This is not a portable way to check if a directory exists
        # After bumping up to 0.53.0, the following line can be replaced with one that uses FS module
        snpe_dir_not_exist = run_command('[', '-d', SNPE_ROOT, ']', check : false).returncode()
        if snpe_dir_not_exist != 0
          error('@0@ does not exists'.format(SNPE_ROOT))
        endif

        snpe_lib = cxx.find_library('libSNPE',
          dirs: join_paths(SNPE_ROOT, 'lib', 'x86_64-linux-clang'),
          required: true
        )

        snpe_incdir = include_directories(join_paths(SNPE_ROOT, 'include', 'zdl'))

        snpe_dep = declare_dependency(
          dependencies: snpe_lib,
          include_directories: snpe_incdir
        )
      endif
    endif
  endif
endif

# tensorrt
nvinfer_dep = dependency('', required: false)
nvparsers_dep = dependency('', required: false)
cuda_dep = dependency('', required: false)
cudart_dep = dependency('', required: false)
if not get_option('tensorrt-support').disabled()
  # check available cuda versions (11.0 and 10.2 are recommended)
  cuda_vers = [
    '11.0',
    '10.2',
    '10.1',
    '10.0',
    '9.2',
    '9.1',
    '9.0'
  ]

  foreach ver : cuda_vers
    cuda_dep = dependency('cuda-' + ver, required: false)
    cudart_dep = dependency('cudart-' + ver, required: false)
    if cuda_dep.found() and cudart_dep.found()
      if ver != '11.0' and ver != '10.2'
        message('Warning: the recommended cuda version is at least 10.2')
      endif
      break
    endif
  endforeach

  nvinfer_lib = cxx.find_library('nvinfer', required: false)
  if nvinfer_lib.found() and cxx.check_header('NvInfer.h')
    nvinfer_dep = declare_dependency(dependencies: nvinfer_lib)
  endif

  nvparsers_lib = cxx.find_library('nvparsers', required: false)
  if nvparsers_lib.found() and cxx.check_header('NvUffParser.h')
    nvparsers_dep = declare_dependency(dependencies: nvparsers_lib)
  endif
endif

# gRPC
grpc_dep = dependency('', required: false)
gpr_dep = dependency('', required: false)
grpcpp_dep = dependency('', required: false)
if not get_option('grpc-support').disabled()
  grpc_dep = dependency('grpc', required: false)
  gpr_dep = dependency('gpr', required: false)
  grpcpp_dep = dependency('grpc++', required: false)
endif

# mqtt (paho.mqtt.c)
pahomqttc_dep = dependency('', required: false)
if not get_option('mqtt-support').disabled()
  pahomqttc_dep = dependency('paho-mqtt-c', required: false)
  if (not pahomqttc_dep.found() and get_option('mqtt-support').enabled())
    message('mqtt-support is enabled while its pkgconfig is not found. Hardcoded build configuration is used.')
    pahomqttc_lib1 = cxx.find_library('paho-mqtt3a', required: true)
    pahomqttc_lib2 = cxx.find_library('paho-mqtt3c', required: true)
    pahomqttc_lib3 = cxx.find_library('paho-mqtt3as', required: true)
    pahomqttc_lib4 = cxx.find_library('paho-mqtt3cs', required: true)
    pahomqttc_dep = declare_dependency(dependencies: [pahomqttc_lib1, pahomqttc_lib2, pahomqttc_lib3, pahomqttc_lib4])
  endif
endif

# deepview-rt
deepview_rt_dep = dependency('', required: false)
if not get_option('deepview-rt-support').disabled()
  deepview_rt_dep = dependency('deepview-rt', required: false)
  if (not deepview_rt_dep.found())
    deepview_rt_lib = cxx.find_library('deepview-rt', required: false)
    if (deepview_rt_lib.found() and cxx.check_header('deepview_rt.h'))
      deepview_rt_dep = declare_dependency(dependencies: [ deepview_rt_lib ])
    endif
  endif
endif
if get_option('deepview-rt-support').enabled() and not deepview_rt_dep.found()
  error('deepview-rt enabled but dependencies are not found.')
endif

# armnn
armnn_dep = dependency('', required: false)
if not get_option('armnn-support').disabled()
  armnn_dep = dependency('armnn', required: false)
  if (not armnn_dep.found())
    armnn_dep = dependency('Armnn', required: false)
  endif
  if (not armnn_dep.found())
    armnn_lib = cxx.find_library('armnn', required: false)
    if (armnn_lib.found() and cxx.check_header('armnn/ArmNN.hpp'))
      armnn_dep = declare_dependency(dependencies: [ armnn_lib, thread_dep ])
    endif
  endif
endif

# mxnet
mxnet_dep = dependency('', required: false)
if not get_option('mxnet-support').disabled()
  mxnet_dep = cxx.find_library('mxnet', required: false)
  if not mxnet_dep.found()
    if cxx.check_header('mxnet-cpp/MxNetCpp.h', required: get_option('mxnet-support'))
      mxnet_dep = declare_dependency(link_args: ['-lmxnet'])
    endif
  endif
endif

# features registration to be controlled
#
# register feature as follows
#  <string: feature_name> :  {
#      target: <string>, extra_deps: <list>,  project_args: <dict>,
#      project_args_disabled: <string>, extra_args: <dict: meson variables to be registered>,
features = {
  'video-support': {
    'project_args_disabled': { 'NO_VIDEO': 1 },
  },
  'audio-support': {
    'project_args_disabled': { 'NO_AUDIO': 1 },
  },
  'tf-support': {
    'target': 'tensorflow',
    'extra_deps': [ protobuf_dep ],
    'project_args': { 'ENABLE_TENSORFLOW': 1 },
  },
  'tflite-support': {
    'target': 'tensorflow-lite',
    'project_args': { 'ENABLE_TENSORFLOW_LITE': 1 }
  },
  'tflite2-support': {
    'target': 'tensorflow2-lite',
    'project_args': { 'ENABLE_TENSORFLOW2_LITE': 1 }
  },
  'pytorch-support': {
    'target': 'pytorch',
    'target_alt': 'Torch',
    'project_args': { 'ENABLE_PYTORCH': 1 }
  },
  'caffe2-support': {
    'target': 'caffe2',
    'project_args': { 'ENABLE_CAFFE2': 1 }
  },
  'deepview-rt-support': {
    'extra_deps': [ deepview_rt_dep ],
    'project_args': { 'ENABLE_DEEPVIEW_RT': 1 }
  },
  'mvncsdk2-support': {
    'target': 'libmvnc',
    'project_args': { 'ENABLE_MOVIDIUS_NCSDK2' : 1}
  },
  'nnfw-runtime-support': {
    'extra_deps': [ nnfw_dep ],
    'project_args': { 'ENABLE_NNFW_RUNTIME': 1 }
  },
  'armnn-support': {
    'extra_deps': [ armnn_dep ],
    'project_args': { 'ENABLE_ARMNN': 1 }
  },
  'orcc-support': {
    'extra_deps': [ orc_dep, pg_orcc ],
    'project_args': {'HAVE_ORC': 1},
    'project_args_disabled': { 'DISABLE_ORC': 1 },
    'extra_args': {'orcc_args': [pg_orcc, '--include', 'glib.h'] }
  },
  'snpe-support': {
    'extra_deps': [ snpe_dep ],
    'project_args': { 'ENABLE_SNPE' : 1 },
  },
  'flatbuf-support': {
    'extra_deps': [ flatc, flatbuf_dep, flatbuf_version_check_dep ],
    'project_args': { 'ENABLE_FLATBUF': 1 }
  },
  'protobuf-support': {
    'extra_deps': [ pb_comp, protobuf_dep ],
    'project_args': { 'ENABLE_PROTOBUF': 1 }
  },
  'tensorrt-support': {
    'extra_deps': [ nvinfer_dep, nvparsers_dep, cuda_dep, cudart_dep ],
    'project_args': { 'ENABLE_TENSORRT': 1 }
  },
  'grpc-support': {
    'extra_deps': [ grpc_dep, gpr_dep, grpcpp_dep ],
    'project_args': { 'ENABLE_GRPC': 1 }
  },
  'lua-support': {
    # TODO: support various Lua versions
    'target': 'lua',
    'target_alt': 'lua5.1',
    'project_args': { 'ENABLE_LUA': 1 }
  },
  'mqtt-support': {
    'extra_deps': [ pahomqttc_dep ],
    'project_args': { 'ENABLE_MQTT': 1 }
  },
  'tvm-support': {
    'target': 'tvm_runtime',
    'project_args': { 'ENABLE_TVM' : 1 }
  },
  'trix-engine-support': {
    'target': 'npu-engine',
    'project_args': { 'ENABLE_TRIX_ENGINE' : 1 }
  },
  'nnstreamer-edge-support': {
    'target': 'nnstreamer-edge',
    'project_args': { 'ENABLE_NNSTREAMER_EDGE': 1 }
  },
  'mxnet-support': {
    'extra_deps': [ mxnet_dep ],
    'project_args': { 'ENABLE_MXNET' : 1 }
  }
}

project_args = {}
# This section controls the flow of feature registration.
foreach feature_name, data : features
  variable_name = feature_name.underscorify()
  variable_deps_name = variable_name + '_deps'
  variable_available_name = variable_name + '_is_available'

  _available = true

  target = data.get('target', '')

  _deps = []

  if target != ''
    target_dep = dependency(target, required: false)
    if not target_dep.found()
      target_alt = data.get('target_alt', '')
      if target_alt != ''
        target_dep = dependency(target_alt, required: false)
      endif
    endif

    if get_option(feature_name).enabled() and not target_dep.found()
      error('@0@ is enabled but unable to find the target dependency'.format(feature_name))
    endif

    _deps += target_dep
  endif
  _deps += data.get('extra_deps', [])


  foreach dep : _deps
    if not dep.found()
      _available = false
    endif
  endforeach

  if get_option(feature_name).disabled() or not _available
    project_args += data.get('project_args_disabled', {})
    set_variable(variable_deps_name, [])
    set_variable(variable_available_name, false)
    message('@0@ is off because it is either not available or disabled'.format(feature_name))
    continue
  endif

  # handle when available
  project_args += data.get('project_args', {})

  set_variable(variable_deps_name, _deps)
  set_variable(variable_available_name, true)

  foreach name, value : data.get('extra_args', {})
    set_variable(variable_name + '_' + name, value)
  endforeach

endforeach

#Definitions enabled by meson_options.txt
message('Following project_args are going to be included')
message(project_args)
foreach name, value: project_args
  add_project_arguments('-D@0@=@1@'.format(name, value), language: ['c', 'cpp'])
endforeach

# Add redundant declaration flag when caffe2 and pytorch both are disabled
if not (pytorch_support_is_available or caffe2_support_is_available)
  redundant_decls_flag = '-Wredundant-decls'
  if cc.has_argument (redundant_decls_flag)
    add_project_arguments([redundant_decls_flag], language: 'c')
  endif
  if cxx.has_argument (redundant_decls_flag)
    add_project_arguments([redundant_decls_flag], language: 'cpp')
  endif
endif

# Python3
have_python3 = false
if not get_option('python3-support').disabled()
  # Check python 3.x
  python3_dep = dependency('python3', required: false)
  if python3_dep.found() and python3_dep.version().version_compare('>= 3.8')
    # The name of .pc file provides C/CXX/LD_FLAGS for Python C API has been changed since v3.8
    python3_dep = dependency('python3-embed', required: false)
  endif

  if python3_dep.found()
    pg_pkgconfig = find_program('pkg-config')

    python3_inc_args = []
    python3_inc_args += run_command(pg_pkgconfig, ['python3', '--cflags'], check : true).stdout().strip().split()
    python3_inc_args += run_command('python3', ['-c', 'import site\nfor i in site.getsitepackages(): print("-I" + i + "/numpy/core/include")'], check : true).stdout().strip().split()
    r = run_command('python3', ['-m', 'site', '--user-site'], check : false)
    if r.returncode() == 0
      python3_inc_args += '-I' + r.stdout().strip() + '/numpy/core/include'
    endif
    python3_inc_valid_args = []

    foreach python3_inc_arg : python3_inc_args
      if cxx.has_argument(python3_inc_arg) and \
          cxx.check_header('numpy/arrayobject.h', args : python3_inc_arg, dependencies : python3_dep)
        python3_inc_valid_args += python3_inc_arg
        have_python3 = true
        break
      endif
    endforeach

    if have_python3
      python3_dep = declare_dependency(dependencies: python3_dep,
          compile_args : python3_inc_valid_args)
    else
      warning('Found python3, but failed to find numpy.')
      warning('Disable nnstreamer-python3.')
      python3_dep = disabler()
    endif
  endif

  if get_option('python3-support').enabled() and not have_python3
    error('Cannot find python3 with numpy support')
  endif
endif

# Set sub-plugin priority
tflite_subplugin_list = []
if tflite2_support_is_available
  tflite_subplugin_list += 'tensorflow2-lite'
endif
if tflite_support_is_available
  tflite_subplugin_list += 'tensorflow1-lite'
endif

nnstreamer_conf.set('TFLITE_SUBPLUGIN_PRIORITY', ','.join(tflite_subplugin_list))

# Set configuration to install .ini
nnstreamer_install_conf = configuration_data()
nnstreamer_install_conf.merge_from(nnstreamer_conf)

nnstreamer_install_conf.set('ENABLE_ENV_VAR', get_option('enable-env-var'))
nnstreamer_install_conf.set('ENABLE_SYMBOLIC_LINK', get_option('enable-symbolic-link'))
nnstreamer_install_conf.set('TORCH_USE_GPU', get_option('enable-pytorch-use-gpu'))

# Element restriction
restriction_config = ''

if get_option('enable-element-restriction')
  restriction_config = '''[element-restriction]
enable_element_restriction=True
allowed_elements=''' + get_option('allowed-elements')
endif

nnstreamer_install_conf.set('ELEMENT_RESTRICTION_CONFIG', restriction_config)
nnstreamer_install_conf.set('TEST_TEMPLATE_DIR', unittest_base_dir) # TEST_TEMPLATE_DIR will be empty if `install-test` is false

# Extra configuration
extra_config_path = ''
if get_option('extra_config_path') != ''
  extra_config_path = 'extra_config_path=' + get_option('extra_config_path')
endif
nnstreamer_install_conf.set('EXTRA_CONFIG_PATH', extra_config_path)

# Install .ini
configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer.ini',
  install_dir: nnstreamer_inidir,
  configuration: nnstreamer_install_conf
)

# Install .pc
configure_file(input: 'nnstreamer-single.pc.in', output: 'nnstreamer-single.pc',
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_install_conf
)
configure_file(input: 'nnstreamer.pc.in', output: 'nnstreamer.pc',
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_install_conf
)
configure_file(input: 'nnstreamer-internal.pc.in', output: 'nnstreamer-internal.pc',
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_install_conf
)

# Check whether mqtt broker is running or not.
check_mosquitto = run_command ('bash', './tests/check_broker.sh', check : true).stdout()
if check_mosquitto != ''
  add_project_arguments('-D__MQTT_BROKER_ENABLED__=1', language: ['c', 'cpp'])
endif

# Float16 Support
if get_option('enable-float16')
  arch = target_machine.cpu_family()
  add_project_arguments('-DFLOAT16_SUPPORT', language: ['c', 'cpp'])

  if arch == 'aarch64'
    message ('Float16 for aarch64 enabled. Modern gcc-aarch64 generally supports float16 with __fp16. It uses IEEE 754-2008 format.')
  elif arch == 'arm'
    add_project_arguments('-mfp16-format=ieee', language: ['c', 'cpp'])
    message ('Float16 for arm (32b) enabled. Modern gcc-arm generally supports float16 with __fp16 by -mfp16-format=ieee for IEEE 754-2008 format.')
  elif arch == 'x86_64' or arch == 'x86'
    # GCC 12+ has fp16 avx512 support.
    has_avx512fp16 = cc.has_argument('-mavx512fp16')

    if (has_avx512fp16)
      add_project_arguments(['-mavx512fp16'], language: ['c', 'cpp'])
      message ('Float16 for x86_64 enabled. Modern gcc-x64 genrally supports float16 with _Float16. -mavx512fp16 added for hardware acceleration')
    else
      warning ('Float16 for x86_64 enabled. However, software emulation is applied for fp16, making it slower and inconsistent. Use GCC 12+ for AVX512 FP16 support. This build will probably fail unless you bring a compiler that supports fp16 for x64.')
    endif
  else
    error ('Float16 is supported only in aarch64, arm, and x86_64. If you believe the compiler for this architecture supports float16, you will need to update the build script (meson.build).')
  endif

endif

# A few distros do not have execinfo.h
if not cc.has_header('execinfo.h')
  add_project_arguments('-D_NO_EXECINFO_', language: ['c', 'cpp'])
endif

# See if src-iio can be built or not
gst18_dep = dependency('gstreamer-' + gst_api_verision, version : '>=1.8', required : false)
tensor_src_iio_build = false
if gst18_dep.found() and build_platform != 'macos'
  add_project_arguments('-D_ENABLE_SRC_IIO', language: ['c', 'cpp'])
  message('tensor_src_iio enabled')
  tensor_src_iio_build = true
else
  message('tensor_src_iio disabled: it requires GStreamer >= 1.8 and Linux')
endif

# Build nnstreamer (common, plugins)
subdir('gst')

# Build ext subplugins
subdir('ext')

# Build Utility
subdir('tools/development')

# Build unittests
if get_option('enable-test')
  # ini file generator template for the plugins from other repository
  configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer-test.ini.in',
    install: get_option('install-test'),
    install_dir: unittest_base_dir,
    copy: true,
  )

  # temporary ini file for test, enable env variables.
  nnstreamer_test_conf = configuration_data()
  nnstreamer_test_conf.merge_from(nnstreamer_conf)

  nnstreamer_test_conf.set('ENABLE_ENV_VAR', true)
  nnstreamer_test_conf.set('ENABLE_SYMBOLIC_LINK', false)
  nnstreamer_test_conf.set('TORCH_USE_GPU', false)
  nnstreamer_test_conf.set('EXTRA_CONFIG_PATH', '')
  nnstreamer_test_conf.set('ELEMENT_RESTRICTION_CONFIG', '')

  configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer-test.ini',
    install: get_option('install-test'),
    install_dir: unittest_base_dir,
    configuration: nnstreamer_test_conf
  )

  path_gst_plugin = join_paths(meson.build_root(), 'gst')
  path_ext_plugin = join_paths(meson.build_root(), 'ext')

  path_nns_conf = join_paths(meson.build_root(), 'nnstreamer-test.ini')
  path_nns_plugin_prefix = join_paths(path_ext_plugin, 'nnstreamer')
  path_nns_plugin_filters = join_paths(path_nns_plugin_prefix, 'tensor_filter')
  path_nns_plugin_decoders = join_paths(path_nns_plugin_prefix, 'tensor_decoder')
  path_nns_plugin_converters = join_paths(path_nns_plugin_prefix, 'tensor_converter')

  testenv = environment()
  testenv.set('GST_PLUGIN_PATH', path_gst_plugin + ':' + path_ext_plugin)
  testenv.set('NNSTREAMER_CONF', path_nns_conf)
  testenv.set('NNSTREAMER_FILTERS', path_nns_plugin_filters)
  testenv.set('NNSTREAMER_DECODERS', path_nns_plugin_decoders)
  testenv.set('NNSTREAMER_CONVERTERS', path_nns_plugin_converters)
  testenv.set('NNSTREAMER_SOURCE_ROOT_PATH', meson.source_root())
  testenv.set('NNSTREAMER_BUILD_ROOT_PATH', meson.build_root())

  subdir('tests')
endif
