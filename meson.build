# If you are using Ubuntu/Xenial, Do "force-version" on meson to get the required version.
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
  version: '0.3.0',
  license: ['LGPL'],
  meson_version: '>=0.42.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++11'
  ]
)

add_project_arguments('-DVERSION="' + meson.project_version() + '"', language: ['c', 'cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')
build_platform = ''

if get_option('enable-tizen')
  # Pass __TIZEN__ to the compiler
  add_project_arguments('-D__TIZEN__=1', language: ['c', 'cpp'])
  build_platform = 'tizen'
elif not meson.is_cross_build()
  if cc.get_id() == 'clang' and cxx.get_id() == 'clang'
    if build_machine.system() == 'darwin'
      # Pass __MACOS__ to the compiler
      add_project_arguments('-D__MACOS__=1', language: ['c', 'cpp'])
      build_platform = 'macos'
    endif
  endif
endif

warning_flags = [
  '-Wredundant-decls',
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
# Work around the issue related to gtest cflags
if meson.version().version_compare('<0.46')
  warning_flags += '-Wmissing-include-dirs'
endif

warning_c_flags = [
  '-Wmissing-declarations',
  '-Wmissing-prototypes',
  '-Wnested-externs',
  '-Waggregate-return',
  '-Wold-style-definition',
  '-Wdeclaration-after-statement'
]

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

# Define default conf file
add_project_arguments('-DNNSTREAMER_CONF_FILE="' + join_paths(nnstreamer_inidir, 'nnstreamer.ini') + '"', language: 'c')

# Dependencies
glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
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

# Orc
have_orcc = false

if get_option('enable-orc')
  orc_dep = dependency('orc-0.4', version: '>= 0.4.17', required: true)
  orcc = find_program('orcc', required: true)
  if orc_dep.found() and orcc.found()
    have_orcc = true
    orcc_args = [orcc, '--include', 'glib.h']
    add_project_arguments('-DHAVE_ORC=1', language: ['c', 'cpp'])
  else
    error('Cannot find orc library')
  endif
else
  add_project_arguments('-DDISABLE_ORC=1', language: ['c', 'cpp'])
endif

# NO Video support
if get_option('disable-video-support')
  add_project_arguments('-DNO_VIDEO=1', language: ['c', 'cpp'])
  message('Disable Video Type Support')
endif

# NO Audio support
if get_option('disable-audio-support')
  add_project_arguments('-DNO_AUDIO=1', language: ['c', 'cpp'])
  message('Disable Audio Type Support')
endif

# Tensorflow
if get_option('enable-tensorflow')
  tf_dep = dependency('tensorflow', required: true)

  if tf_dep.found() and protobuf_dep.found()
    add_project_arguments('-DENABLE_TENSORFLOW=1', language: ['c', 'cpp'])
  else
    error('Cannot find tensorflow')
  endif
endif

# Tensorflow-lite
if get_option('enable-tensorflow-lite')
  tflite_dep = dependency('tensorflow-lite', required: true)

  if tflite_dep.found()
    add_project_arguments('-DENABLE_TENSORFLOW_LITE=1', language: ['c', 'cpp'])
  else
    error('Cannot find tensorflow-lite')
  endif
endif

# PyTorch
if get_option('enable-pytorch')
  torch_dep = dependency('pytorch', required: true)

  if torch_dep.found()
    add_project_arguments('-DENABLE_PYTORCH=1', language: ['c', 'cpp'])
  else
    error('Cannot find pytorch')
  endif
endif

# Caffe2
if get_option('enable-caffe2')
  caffe2_dep = dependency('caffe2', required: true)

  if caffe2_dep.found() and protobuf_dep.found()
    add_project_arguments('-DENABLE_CAFFE2=1', language: ['c', 'cpp'])
  else
    error('Cannot find caffe2')
  endif
endif

# Python
have_python2 = false
have_python3 = false

if get_option('enable-python')
  pg_pkgconfig = find_program('pkg-config')

  # Check python 2.7
  python2_dep = dependency('python-2.7', required: false)
  if python2_dep.found()
    python2_incs = run_command(pg_pkgconfig, ['python-2.7', '--cflags']).stdout().strip().split()
    if cc.has_header('numpy/arrayobject.h', args: python2_incs)
      have_python2 = true
    endif
  endif

  # Check python 3.x
  python3_dep = dependency('python3', required: false)
  if python3_dep.found()
    python3_incs = run_command(pg_pkgconfig, ['python3', '--cflags']).stdout().strip().split()
    if cc.has_header('numpy/arrayobject.h', args: python3_incs)
      have_python3 = true
    endif
  endif
endif

# nnfw ( details in https://review.tizen.org/gerrit/p/platform/core/ml/nnfw )
if get_option('enable-nnfw')
  add_project_arguments('-DENABLE_NNFW=1', language: ['c', 'cpp'])
endif

# Patch for non-tizen build
patch_file = join_paths(meson.current_source_dir(), 'packaging', 'non_tizen_build.patch')

# Build nnstreamer (common, plugins)
subdir('gst')

# Build ext subplugins
subdir('ext')

# Build API
subdir('api')

# Build nnstreamer examples
if get_option('enable-test') or get_option('install-example')
  subdir('nnstreamer_example')
endif

# Build unittests
if get_option('enable-test')
  subdir('tests')

  # temporary ini file for test, enable env variables.
  nnstreamer_test_conf = configuration_data()
  nnstreamer_test_conf.merge_from(nnstreamer_conf)

  nnstreamer_test_conf.set('ENABLE_ENV_VAR', true)
  nnstreamer_test_conf.set('ENABLE_SYMBOLIC_LINK', false)
  nnstreamer_test_conf.set('TORCH_USE_GPU', false)
  nnstreamer_test_conf.set('ELEMENT_RESTRICTION_CONFIG', '')

  # meson 0.50 supports install argument in configure_file()
  if get_option('install-test')
    configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer-test.ini',
      install_dir: unittest_install_dir,
      configuration: nnstreamer_test_conf
    )
  else
    configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer-test.ini',
      configuration: nnstreamer_test_conf
    )
  endif
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
