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
    dependencies: [nnstreamer_dep, glib_dep, gst_dep],
    install: true,
    install_dir: converter_subplugin_install_dir
  )
  static_library('nnstreamer_converter_flatbuf',
    nnstreamer_converter_flatbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep],
    install: true,
    install_dir: nnstreamer_libdir
  ) 
endif
