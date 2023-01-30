gstrepo_src_inc = include_directories('.', '../nnstreamer/include/')

repo_src_sources = [
  'gstrepo_src.c',
]

gstrepo_src_shared = shared_library('gstrepo_src',
  repo_src_sources,
  dependencies: [glib_dep, gst_dep, gst_base_dep],
  include_directories: gstrepo_src_inc,
  install: true,
  install_dir: plugins_install_dir
)

gstrepo_src_static = static_library('gstrepo_src',
  repo_src_sources,
  dependencies: [glib_dep, gst_dep, gst_base_dep],
  include_directories: gstrepo_src_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

gstrepo_src_lib = gstrepo_src_shared
if get_option('default_library') == 'static'
  gstrepo_src_lib = gstrepo_src_static
endif

repo_src_dep = declare_dependency(link_with: gstrepo_src_lib)
