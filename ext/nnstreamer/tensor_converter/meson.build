# flatbuffer
if flatbuf_support_is_available
  converter_sub_flatbuf_sources = [
    'tensor_converter_flatbuf.cc'
  ]

  nnstreamer_converter_flatbuf_sources = [fb_gen_src]
  foreach s : converter_sub_flatbuf_sources
    nnstreamer_converter_flatbuf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  shared_library('nnstreamer_converter_flatbuf',
    nnstreamer_converter_flatbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )

  converter_sub_flexbuf_sources = [
    'tensor_converter_flexbuf.cc'
  ]

  nnstreamer_converter_flexbuf_sources = []
  foreach s : converter_sub_flexbuf_sources
    nnstreamer_converter_flexbuf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  shared_library('nnstreamer_converter_flexbuf',
    nnstreamer_converter_flexbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif

# protocol buffer
if protobuf_support_is_available
  converter_sub_protobuf_sources = [
    'tensor_converter_protobuf.cc'
  ]

  nnstreamer_converter_protobuf_sources = []
  foreach s : converter_sub_protobuf_sources
    nnstreamer_converter_protobuf_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  shared_library('nnstreamer_converter_protobuf',
    nnstreamer_converter_protobuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, protobuf_util_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif
