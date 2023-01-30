gstdatarepo_src_inc = include_directories('.', '../nnstreamer/include/')

repo_sources = [
  'datarepo_elements.c',
  'gstdatareposrc.c'
]

gstdatarepo_shared = shared_library('gstdatarepo',
  repo_sources,
  dependencies: [glib_dep, gst_dep, gst_base_dep],
  include_directories: gstdatarepo_src_inc,
  install: true,
  install_dir: plugins_install_dir
)

gstdatarepo_static = static_library('gstdatarepo',
  repo_sources,
  dependencies: [glib_dep, gst_dep, gst_base_dep],
  include_directories: gstdatarepo_src_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

gstdatarepo_lib = gstdatarepo_shared
if get_option('default_library') == 'static'
  gstdatarepo_lib = gstdatarepo_static
endif

repo_dep = declare_dependency(link_with: gstdatarepo_lib)
