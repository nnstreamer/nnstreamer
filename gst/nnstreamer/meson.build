nnstreamer_inc = include_directories('.', './include/')
nnstreamer_headers = []

# Dependencies
nnstreamer_single_deps = [
  glib_dep,
  gmodule_dep,
  gobject_dep
]

# log utils
# TODO: Define proper variable for android in entire project
if build_platform == 'tizen'
  nnstreamer_single_deps += dlog_dep
elif cc.has_header_symbol('android/log.h', '__android_log_print')
  nnstreamer_single_deps += cc.find_library('log')
endif

nnstreamer_deps = [
  libm_dep,
  gst_dep,
  gst_base_dep,
  gst_controller_dep,
  gst_video_dep,
  gst_audio_dep,
  thread_dep,
  gio_dep
]

if orcc_support_is_available
  nnstreamer_deps += orc_dep
endif

if nnstreamer_edge_support_is_available
  nnstreamer_deps += nnstreamer_edge_support_deps
endif

# Internal dependencies
nnstreamer_internal_deps = []

# Add nnstreamer single sources
nnstreamer_single_sources = [
  'hw_accel.c',
  'nnstreamer_conf.c',
  'nnstreamer_log.c',
  'nnstreamer_subplugin.c',
  'nnstreamer_plugin_api_util_impl.c'
]

# Add nnstreamer registerer and common sources
nnstreamer_sources = [
  'registerer/nnstreamer.c',
  'nnstreamer_plugin_api_impl.c',
  'tensor_allocator.c',
  'tensor_data.c',
  'tensor_meta.c'
]

# Add plugins
nnst_plugins = [
  'tensor_filter',
  'tensor_query',
  'elements'
]

foreach p : nnst_plugins
  subdir(p)
endforeach

subdir('include')

# Private header for sub-plugins and native APIs
nnstreamer_headers += files('nnstreamer_internal.h')

nnstreamer_single_shared = shared_library('nnstreamer-single',
  nnstreamer_single_sources,
  dependencies: nnstreamer_single_deps,
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

nnstreamer_single_static = static_library('nnstreamer-single',
  nnstreamer_single_sources,
  dependencies: nnstreamer_single_deps,
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

nnstreamer_single_lib = nnstreamer_single_shared
if get_option('default_library') == 'static'
  nnstreamer_single_lib = nnstreamer_single_static
endif

nnstreamer_single_dep = declare_dependency(link_with: nnstreamer_single_lib,
  dependencies: nnstreamer_single_deps,
  include_directories: nnstreamer_inc)

nnstreamer_deps += nnstreamer_single_dep

# Build libraries ("both_libraries" are supported from 0.46.)
nnstreamer_shared = shared_library('nnstreamer',
  nnstreamer_sources,
  dependencies: nnstreamer_deps,
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: plugins_install_dir
)

## @todo target name using method name() (since 0.54.0)
# target_name = nnstreamer_shared.name()
libpath_split = nnstreamer_shared.full_path().split('/')
target_name = libpath_split[libpath_split.length() - 1]

meson.add_install_script(
  'sh', '-c',
  'ln -fs @0@ @1@@2@'.format(
    join_paths(plugins_install_dir, target_name),
    '${DESTDIR}',
    join_paths(nnstreamer_libdir, target_name)))

nnstreamer_static = static_library('nnstreamer',
  nnstreamer_sources,
  dependencies: nnstreamer_deps,
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

nnstreamer_lib = nnstreamer_shared
if get_option('default_library') == 'static'
  nnstreamer_lib = nnstreamer_static
endif

nnstreamer_dep = declare_dependency(link_with: nnstreamer_lib,
  dependencies: nnstreamer_deps,
  include_directories: nnstreamer_inc)

# Install headers into /{includedir}/nnstreamer
install_headers(nnstreamer_headers,
  subdir: 'nnstreamer'
)
