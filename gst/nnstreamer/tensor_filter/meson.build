tensor_filter_sources = [
  'tensor_filter.c',
  'tensor_filter_common.c',
  'tensor_filter_custom.c',
  'tensor_filter_custom_easy.c'
]

foreach s : tensor_filter_sources
  nnstreamer_sources += join_paths(meson.current_source_dir(), s)
endforeach

if get_option('enable-filter-cpp-class')
  nnstreamer_sources += join_paths(meson.current_source_dir(), 'tensor_filter_support_cc.cc')
endif

# filter single-shot
nnstreamer_sources += join_paths(meson.current_source_dir(), 'tensor_filter_single.c')
nnstreamer_headers += join_paths(meson.current_source_dir(), 'tensor_filter_single.h')
