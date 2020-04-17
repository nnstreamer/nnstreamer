# Common headers to be installed
nnst_common_headers = [
  'tensor_typedef.h',
  'tensor_filter_custom.h',
  'tensor_filter_custom_easy.h',
  'nnstreamer_plugin_api_filter.h',
  'nnstreamer_plugin_api_decoder.h',
  'nnstreamer_plugin_api_converter.h',
  'nnstreamer_plugin_api.h'
]

foreach h : nnst_common_headers
  nnstreamer_headers += join_paths(meson.current_source_dir(), h)
endforeach

# Install headers into /{includedir}/nnstreamer
install_headers(nnstreamer_headers,
  subdir: 'nnstreamer'
)

if get_option('enable-filter-cpp-class')
  install_headers(['nnstreamer_cppplugin_api_filter.hh'],
    subdir: 'nnstreamer'
  )
endif
