# If you are using Ubuntu/Xenial, Do "force-version" on meson to get 0.40
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
  version: '0.1.2',
  license: ['LGPL'],
  meson_version: '>=0.40.0',
  default_options: [
    'werror=true',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++11'
  ]
)

add_project_arguments('-DVERSION="'+meson.project_version()+'"', language: ['c', 'cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

gst_api_verision = '1.0'

# Set configuration
nnstreamer_conf = configuration_data()
nnstreamer_conf.set('VERSION', meson.project_version())

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

nnstreamer_conf.set('PREFIX', nnstreamer_prefix)
nnstreamer_conf.set('EXEC_PREFIX', nnstreamer_bindir)
nnstreamer_conf.set('LIB_INSTALL_DIR', nnstreamer_libdir)
nnstreamer_conf.set('GST_INSTALL_DIR', plugins_install_dir)
nnstreamer_conf.set('INCLUDE_INSTALL_DIR', nnstreamer_includedir)
nnstreamer_conf.set('SUBPLUGIN_INSTALL_PREFIX', subplugin_install_prefix)

# Dependencies
glib_dep = dependency('glib-2.0')
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

# Orc
have_orcc = false

orc_dep = dependency('orc-0.4', version: '>= 0.4.17', required: false)
orcc = find_program('orcc', required: false)
if orc_dep.found() and orcc.found()
  have_orcc = true
  orcc_args = [orcc, '--include', 'glib.h']
  add_project_arguments('-DHAVE_ORC=1', language: ['c', 'cpp'])
else
  message('Cannot find orc library')
  add_project_arguments('-DDISABLE_ORC=1', language: ['c', 'cpp'])
endif

# NO Audio support
disable_audio = false

if get_option('disable-audio-support')
   disable_audio = true
   add_project_arguments('-DNO_AUDIO=1', language: ['c', 'cpp'])
   message('Disable Audio Type Support')
endif

# Tensorflow
have_tensorflow = false

if get_option('enable-tensorflow')
  tf_dep = dependency('tensorflow', required: true)
  protobuf_dep = dependency('protobuf', version: '>= 3.4.0', required: true)

  if tf_dep.found() and protobuf_dep.found()
    have_tensorflow = true
    add_project_arguments('-DENABLE_TENSORFLOW=1', language: ['c', 'cpp'])
  else
    message('Cannot find tensorflow')
  endif

  nnstreamer_conf.set('TF_MEM_OPTMZ', get_option('enable-tensorflow-mem-optmz'))
else
  message('Tensorflow is disabled')
  nnstreamer_conf.set('TF_MEM_OPTMZ', false)
endif

# Tensorflow-lite
have_tensorflow_lite = false

if get_option('enable-tensorflow-lite')
  tflite_dep = dependency('tensorflow-lite', required: true)

  if tflite_dep.found()
    have_tensorflow_lite = true
    add_project_arguments('-DENABLE_TENSORFLOW_LITE=1', language: ['c', 'cpp'])
  else
    message('Cannot find tensorflow-lite')
  endif
endif

# Build nnstreamer (common, plugins)
subdir('gst')

# Build ext subplugins
subdir('ext')

# Build nnstreamer examples
if get_option('enable-test') or get_option('install-example')
  subdir('nnstreamer_example')
endif

# Build unittests
if get_option('enable-test')
  subdir('tests')
endif

# Install .ini
configure_file(input: 'nnstreamer.ini.in', output: 'nnstreamer.ini',
  install: true,
  install_dir: nnstreamer_inidir,
  configuration: nnstreamer_conf
)

# Install .pc
configure_file(input: 'nnstreamer.pc.in', output: 'nnstreamer.pc',
  install: true,
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_conf
)
