if protobuf_support_is_available
  converter_sub_protobuf_sources = [
    'nnstreamer_protobuf.cc'
  ]

  nnstreamer_converter_protobuf_sources = [pb_gen_src]
  foreach s : converter_sub_protobuf_sources
    nnstreamer_converter_protobuf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  protobuf_util_shared = shared_library('nnstreamer_protobuf',
    nnstreamer_converter_protobuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, protobuf_dep],
    install: true,
    install_dir: nnstreamer_libdir
  )

  protobuf_util_dep = declare_dependency(link_with: protobuf_util_shared,
    dependencies: protobuf_dep,
    include_directories: include_directories('.')
  )
endif
