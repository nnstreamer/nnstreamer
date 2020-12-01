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


if grpc_support_is_available
  grpc_tensor_src_source_files = [
    'tensor_src_grpc.c',
  ]
  grpc_tensor_src_sources = []

  foreach s : grpc_tensor_src_source_files
    grpc_tensor_src_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  tensor_src_grpc_dep = declare_dependency(
    sources : grpc_tensor_src_sources,
    dependencies : grpc_util_dep
  )
endif
