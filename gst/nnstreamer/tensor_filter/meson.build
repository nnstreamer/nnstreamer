tensor_filter_single_sources = [
  'tensor_filter_single.c',
  'tensor_filter_common.c',
  'tensor_filter_custom.c',
  'tensor_filter_custom_easy.c'
]

foreach s : tensor_filter_single_sources
  nnstreamer_single_sources += join_paths(meson.current_source_dir(), s)
endforeach
nnstreamer_headers += join_paths(meson.current_source_dir(), 'tensor_filter_single.h')

nnstreamer_sources += join_paths(meson.current_source_dir(), 'tensor_filter.c')

if get_option('enable-filter-cpp-class')
  nnstreamer_sources += join_paths(meson.current_source_dir(), 'tensor_filter_support_cc.cc')
endif

