# To resolve compiler warning. Remove this include after removing unused parameters.
nns_util_inc = include_directories('../nnstreamer/include')

mqtt_plugin_srcs = ['mqttsink.c', 'mqttsrc.c', 'mqttelements.c', 'ntputil.c']

gstmqtt_shared = shared_library('gstmqtt',
  mqtt_plugin_srcs,
  dependencies: [glib_dep, gst_dep, gst_base_dep, thread_dep, pahomqttc_dep],
  install: true,
  include_directories: nns_util_inc,
  install_dir: plugins_install_dir
)

gstmqtt_dep = declare_dependency(link_with: gstmqtt_shared,
    include_directories: include_directories('.'))
