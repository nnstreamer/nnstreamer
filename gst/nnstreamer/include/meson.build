# Common headers to be installed
nnstreamer_headers += files(
  'tensor_if.h',
  'tensor_typedef.h',
  'tensor_filter_custom.h',
  'tensor_filter_custom_easy.h',
  'tensor_converter_custom.h',
  'tensor_decoder_custom.h',
  'nnstreamer_plugin_api_filter.h',
  'nnstreamer_plugin_api_decoder.h',
  'nnstreamer_plugin_api_converter.h',
  'nnstreamer_plugin_api_util.h',
  'nnstreamer_plugin_api.h',
  'nnstreamer_util.h'
)

if get_option('enable-filter-cpp-class')
  nnstreamer_headers += files('nnstreamer_cppplugin_api_filter.hh')
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
