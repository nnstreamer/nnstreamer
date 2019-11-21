
if get_option('enable-tizen-sensor')
  tzn_tensor_src_source_files = ['tensor_src_tizensensor.c']
  tzn_tensor_src_sources = []

  foreach s : tzn_tensor_src_source_files
    tzn_tensor_src_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  tznsensor_dep = dependency('capi-system-sensor', required: true)
  tensor_src_tizensensor_deps = [glib_dep, gst_dep, nnstreamer_dep, tznsensor_dep]

  tensor_src_tizensensor_lib = shared_library('nnstreamer-tizen-sensor',
    tzn_tensor_src_sources,
    dependencies: tensor_src_tizensensor_deps,
    install: true,
    install_dir: plugins_install_dir
  )

  tensor_src_tizensensor_static = static_library('nnstreamer-tizen-sensor',
    tzn_tensor_src_sources,
    dependencies: tensor_src_tizensensor_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
