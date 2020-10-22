if get_option('enable-tizen-sensor')
  tzn_tensor_src_source_files = ['tensor_src_tizensensor.c']
  tzn_tensor_src_sources = []

  foreach s : tzn_tensor_src_source_files
    tzn_tensor_src_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  tznsensor_dep = dependency('capi-system-sensor', required: true)

  tensor_src_tizensensor_dep = declare_dependency(
    sources : tzn_tensor_src_sources,
    dependencies : [glib_dep, gst_dep, nnstreamer_dep, tznsensor_dep]
  )
endif
