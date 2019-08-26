tensor_src_sources = [
  'tensor_src_iio.c'
]

foreach s : tensor_src_sources
  if build_platform != 'macos'
    nnstreamer_sources += join_paths(meson.current_source_dir(), s)
  endif
endforeach
