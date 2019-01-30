tensor_decoder_sources = [
  'tensordec.c',
  'tensordec-directvideo.c',
  'tensordec-imagelabel.c',
  'tensordec-boundingbox.c'
]

foreach s : tensor_decoder_sources
  nnstreamer_sources += join_paths(meson.current_source_dir(), s)
endforeach
