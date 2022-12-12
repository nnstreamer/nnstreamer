nnstreamer_single_sources += files(
  'tensor_filter_single.c',
  'tensor_filter_common.c',
  'tensor_filter_custom.c',
  'tensor_filter_custom_easy.c'
)

nnstreamer_headers += files('tensor_filter_single.h')

nnstreamer_sources += files('tensor_filter.c')

if get_option('enable-filter-cpp-class')
  nnstreamer_sources += files('tensor_filter_support_cc.cc')
endif

