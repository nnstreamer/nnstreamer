# If you are using Ubuntu/Xenial, Do "force-version" on meson to get the required version.
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
  version: '1.7.0',
  license: ['LGPL-2.1'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=true',
    'warning_level=1',
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

  if get_option('enable-tizen-feature-check')
    add_project_arguments('-D__FEATURE_CHECK_SUPPORT__', language: ['c', 'cpp'])
  endif

  if get_option('enable-tizen-privilege-check')
    add_project_arguments('-D__PRIVILEGE_CHECK_SUPPORT__', language: ['c', 'cpp'])
  endif

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

# Setup warning flags for c anc cpp
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
nnstreamer_inidir = get_option('sysconfdir')

# nnstreamer plugins path
plugins_install_dir = join_paths(nnstreamer_libdir, 'gstreamer-' + gst_api_verision)

# nnstreamer sub-plugins path
subplugin_install_prefix = join_paths(nnstreamer_prefix, 'lib', 'nnstreamer')
filter_subplugin_install_dir = join_paths(subplugin_install_prefix, 'filters')
decoder_subplugin_install_dir = join_paths(subplugin_install_prefix, 'decoders')
customfilter_install_dir = join_paths(subplugin_install_prefix, 'customfilters')
converter_subplugin_install_dir = join_paths(subplugin_install_prefix, 'converters')
unittest_install_dir = join_paths(subplugin_install_prefix, 'unittest')

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

# Define default conf file
add_project_arguments('-DNNSTREAMER_CONF_FILE="' + join_paths(nnstreamer_inidir, 'nnstreamer.ini') + '"', language: 'c')

# Dependencies
glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
gmodule_dep = dependency('gmodule-2.0')
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
if flatc.found()
  flatc_ver = run_command(flatc, '--version').stdout().split()[2]
  flatbuf_dep = dependency('flatbuffers', version: flatc_ver,
      required : get_option('flatbuf-support'))
endif

# Protobuf compiler
pb_comp = find_program('protoc', required: get_option('protobuf-support'))

#orc
pg_orcc = find_program('orcc', required: get_option('orcc-support'))
orc_dep = dependency('orc-0.4', version: '>= 0.4.17', required: get_option('orcc-support'))

#movidius
mvncsdk2_dep = dependency('', required: false)
if not get_option('mvncsdk2-support').disabled()
  mvncsdk2_dep = cc.find_library('mvnc', required: false)
  if not mvncsdk2_dep.found()
    if cc.check_header('mvnc2/mvnc.h', required: get_option('mvncsdk2-support'))
      mvncsdk2_dep = declare_dependency()
    endif
  endif
endif

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
SNPE_ROOT = ''
if not get_option('snpe-support').disabled()
  # Check $SNPE_ROOT (where SNPE SDK is located)
  cmd = run_command('sh', '-c', 'echo $SNPE_ROOT')
  SNPE_ROOT = cmd.stdout().strip()
  if SNPE_ROOT != ''
    snpe_dep = declare_dependency()
  endif
endif

# tensorrt
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
  nvinfer_dep = declare_dependency(dependencies: nvinfer_lib)

  tensorrt_deps = [ nvinfer_dep, cuda_dep, cudart_dep ]
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
    'project_args': { 'ENABLE_PYTORCH': 1 }
  },
  'caffe2-support': {
    'target': 'caffe2',
    'project_args': { 'ENABLE_CAFFE2': 1 }
  },
  'mvncsdk2-support': {
    'extra_deps': [ mvncsdk2_dep ],
    'project_args': { 'ENABLE_MOVIDIUS_NCSDK2' : 1}
  },
  'nnfw-runtime-support': {
    'extra_deps': [ nnfw_dep ],
    'project_args': { 'ENABLE_NNFW_RUNTIME': 1 }
  },
  'armnn-support': {
    'target': 'armnn',
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
    'extra_args': { 'SNPE_ROOT': SNPE_ROOT }
  },
  'flatbuf-support': {
    'extra_deps': [ flatc, flatbuf_dep ],
    'project_args': { 'ENABLE_FLATBUF': 1 }
  },
  'protobuf-support': {
    'extra_deps': [ pb_comp ]
  },
  'tensorrt-support': {
    'extra_deps': tensorrt_deps
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

  _deps = data.get('extra_deps', [])

  if target != ''
    _deps += dependency(target, required: get_option(feature_name))
  endif

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
if not (pytorch_support_is_available and caffe2_support_is_available)
  redundant_decls_flag = '-Wredundant-decls'
  if cc.has_argument (redundant_decls_flag)
    add_project_arguments([redundant_decls_flag], language: 'c')
  endif
  if cxx.has_argument (redundant_decls_flag)
    add_project_arguments([redundant_decls_flag], language: 'cpp')
  endif
endif

# Python
have_python2 = false
have_python3 = false

pg_pkgconfig = find_program('pkg-config')

# Check python 2.7
python2_dep = dependency('python-2.7', required: get_option('python2-support'))
if python2_dep.found()

  python2_inc_args = []
  python2_inc_args += run_command(pg_pkgconfig, ['python-2.7', '--cflags']).stdout().strip().split()
  python2_inc_args += run_command('python2.7', ['-c', 'import site\nfor i in site.getsitepackages(): print("-I" + i + "/numpy/core/include")']).stdout().strip().split()
  python2_inc_args += '-I' + run_command('python2.7', ['-m', 'site', '--user-site']).stdout().strip() + '/numpy/core/include'
  python2_inc_valid_args = []

  foreach python2_inc_arg : python2_inc_args
    if cxx.has_argument(python2_inc_arg) and \
        cxx.check_header('numpy/arrayobject.h', args: python2_inc_arg, dependencies : python2_dep)
      python2_inc_valid_args += python2_inc_arg
      have_python2 = true
      break
    endif
  endforeach

  if have_python2
    python2_deps = declare_dependency(dependencies: python2_dep,
        compile_args : python2_inc_valid_args)
  else
    warning('Found python2.7, but failed to find numpy.')
    warning('Disable nnstreamer-python2.')
  endif

else
  warning('Python2.7 either not found or disabled')

endif

# Check python 3.x
python3_dep = dependency('python3', required: get_option('python3-support'))
if python3_dep.found()
  if python3_dep.version().version_compare('>= 3.8')
    # The name of .pc file provides C/CXX/LD_FLAGS for Python C API has been changed since v3.8
    python3_dep = dependency('python3-embed', required: get_option('python3-support'))
  endif
  python3_inc_args = []
  python3_inc_args += run_command(pg_pkgconfig, ['python3', '--cflags']).stdout().strip().split()
  python3_inc_args += run_command('python3', ['-c', 'import site\nfor i in site.getsitepackages(): print("-I" + i + "/numpy/core/include")']).stdout().strip().split()
  python3_inc_args += '-I' + run_command('python3', ['-m', 'site', '--user-site']).stdout().strip() + '/numpy/core/include'
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
    python3_deps = declare_dependency(dependencies: python3_dep,
        compile_args : python3_inc_valid_args)
  else
    warning('Found python3, but failed to find numpy.')
    warning('Disable nnstreamer-python3.')
  endif
else
  warning('python3 either not found or disabled')
endif

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
restricted_elements=''' + get_option('restricted-elements')
endif

nnstreamer_install_conf.set('ELEMENT_RESTRICTION_CONFIG', restriction_config)

# Install .ini
configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer.ini',
  install_dir: nnstreamer_inidir,
  configuration: nnstreamer_install_conf
)

# Install .pc
configure_file(input: 'nnstreamer.pc.in', output: 'nnstreamer.pc',
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_install_conf
)

# Build nnstreamer (common, plugins)
subdir('gst')

# Build ext subplugins
subdir('ext')

# Build API
subdir('api')

# Build unittests
if get_option('enable-test')
  # Build nnstreamer examples
  subdir('nnstreamer_example')

  # temporary ini file for test, enable env variables.
  nnstreamer_test_conf = configuration_data()
  nnstreamer_test_conf.merge_from(nnstreamer_conf)

  nnstreamer_test_conf.set('ENABLE_ENV_VAR', true)
  nnstreamer_test_conf.set('ENABLE_SYMBOLIC_LINK', false)
  nnstreamer_test_conf.set('TORCH_USE_GPU', false)
  nnstreamer_test_conf.set('ELEMENT_RESTRICTION_CONFIG', '')

  configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer-test.ini',
    install: get_option('install-test'),
    install_dir: unittest_install_dir,
    configuration: nnstreamer_test_conf
  )

  path_gst_plugin = join_paths(meson.build_root(), 'gst', 'nnstreamer')
  path_nns_conf = join_paths(meson.build_root(), 'nnstreamer-test.ini')
  path_nns_plugin_prefix = join_paths(meson.build_root(), 'ext', 'nnstreamer')
  path_nns_plugin_filters = join_paths(path_nns_plugin_prefix, 'tensor_filter')
  path_nns_plugin_decoders = join_paths(path_nns_plugin_prefix, 'tensor_decoder')
  path_nns_plugin_converters = join_paths(path_nns_plugin_prefix, 'tensor_converter')

  testenv = environment()
  testenv.set('GST_PLUGIN_PATH', path_gst_plugin + ':' + path_nns_plugin_prefix)
  testenv.set('NNSTREAMER_CONF', path_nns_conf)
  testenv.set('NNSTREAMER_FILTERS', path_nns_plugin_filters)
  testenv.set('NNSTREAMER_DECODERS', path_nns_plugin_decoders)
  testenv.set('NNSTREAMER_CONVERTERS', path_nns_plugin_converters)
  testenv.set('NNSTREAMER_SOURCE_ROOT_PATH', meson.source_root())
  testenv.set('NNSTREAMER_BUILD_ROOT_PATH', meson.build_root())

  subdir('tests')
endif
