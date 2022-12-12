# flatbuffer
if flatbuf_support_is_available
  converter_sub_flatbuf_sources = [
    'tensor_converter_flatbuf.cc',
    'tensor_converter_util.c']

  converter_sub_flatbuf_sources += fb_gen_src

  shared_library('nnstreamer_converter_flatbuf',
    converter_sub_flatbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )

  converter_sub_flexbuf_sources = [
    'tensor_converter_flexbuf.cc',
    'tensor_converter_util.c'
  ]

  shared_library('nnstreamer_converter_flexbuf',
    converter_sub_flexbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif

# protocol buffer
if protobuf_support_is_available
  converter_sub_protobuf_sources = [
    'tensor_converter_protobuf.cc',
    'tensor_converter_util.c'
  ]

  shared_library('nnstreamer_converter_protobuf',
    converter_sub_protobuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, protobuf_util_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif

# python3
if have_python3
  converter_sub_python3_sources = ['tensor_converter_python3.cc']

  shared_library('nnstreamer_converter_python3',
    converter_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: converter_subplugin_install_dir
  )

  static_library('nnstreamer_converter_python3',
    converter_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
