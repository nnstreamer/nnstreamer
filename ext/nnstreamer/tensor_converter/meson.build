# flatbuffer
converter_sub_flatbuf_sources = [
  'tensor_converter_flatbuf.c'
]

nnstreamer_converter_flatbuf_sources = []
foreach s : converter_sub_flatbuf_sources
  nnstreamer_converter_flatbuf_sources += join_paths(meson.current_source_dir(), s)
endforeach

shared_library('nnstreamer_converter_flatbuf',
  nnstreamer_converter_flatbuf_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_converter_flatbuf',
  nnstreamer_converter_flatbuf_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)