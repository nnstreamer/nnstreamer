# Common converter util
converter_util_sources = files('tensor_converter_util.c')
converter_util_deps = [nnstreamer_dep, glib_dep, gst_dep]

converter_util_shared = shared_library('nnstreamer-converter-util',
  converter_util_sources,
  dependencies: converter_util_deps,
  install: true,
  install_dir: nnstreamer_libdir
)
converter_util_shared_dep = declare_dependency(
  link_with: converter_util_shared,
  dependencies: converter_util_deps
)

converter_util_static = static_library('nnstreamer-converter-util',
  converter_util_sources,
  dependencies: converter_util_deps,
  install: true,
  install_dir: nnstreamer_libdir
)
converter_util_static_dep = declare_dependency(
  link_with: converter_util_static,
  dependencies: converter_util_deps
)

# flatbuffer
if flatbuf_support_is_available
  converter_sub_flatbuf_sources = files('tensor_converter_flatbuf.cc')

  shared_library('nnstreamer_converter_flatbuf',
    converter_sub_flatbuf_sources,
    dependencies: [converter_util_shared_dep, flatbuf_util_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )

  converter_sub_flexbuf_sources = files('tensor_converter_flexbuf.cc')

  shared_library('nnstreamer_converter_flexbuf',
    converter_sub_flexbuf_sources,
    dependencies: [converter_util_shared_dep, flatbuf_util_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif

# protocol buffer
if protobuf_support_is_available
  converter_sub_protobuf_sources = files('tensor_converter_protobuf.cc')

  shared_library('nnstreamer_converter_protobuf',
    converter_sub_protobuf_sources,
    dependencies: [converter_util_shared_dep, protobuf_util_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
endif

# python3
if have_python3
  converter_sub_python3_sources = files('tensor_converter_python3.cc')

  shared_library('nnstreamer_converter_python3',
    converter_sub_python3_sources,
    dependencies: [converter_util_shared_dep, nnstreamer_python3_helper_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )

  static_library('nnstreamer_converter_python3',
    converter_sub_python3_sources,
    dependencies: [converter_util_static_dep, nnstreamer_python3_helper_dep],
    install: true,
    install_dir: nnstreamer_libdir
  )
endif
