# Common headers to be installed
nnst_common_headers = [
  'tensor_if.h',
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

if get_option('enable-filter-cpp-class')
  nnstreamer_headers += join_paths(meson.current_source_dir(), 'nnstreamer_cppplugin_api_filter.hh')
endif

version_conf = configuration_data()
version_conf.set('__NNSTREAMER_VERSION_MAJOR__', version_split[0])
version_conf.set('__NNSTREAMER_VERSION_MINOR__', version_split[1])
version_conf.set('__NNSTREAMER_VERSION_MICRO__', version_split[2])

generated_h = configure_file(input: 'nnstreamer_version.h.in',
  output: 'nnstreamer_version.h',
  configuration: version_conf
)
nnstreamer_headers += generated_h
