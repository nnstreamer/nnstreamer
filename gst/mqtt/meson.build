mqtt_plugin_srcs = [
  join_paths(meson.current_source_dir(), 'mqttsink.c'),
  join_paths(meson.current_source_dir(), 'mqttsrc.c'),
  join_paths(meson.current_source_dir(), 'mqttelements.c'),
]

gstmqtt_shared = shared_library('gstmqtt',
  mqtt_plugin_srcs,
  dependencies: [glib_dep, gst_dep, gst_base_dep, pahomqttc_dep],
  install: false,
  install_dir: plugins_install_dir
)

gstmqtt_static = static_library('gstmqtt',
  mqtt_plugin_srcs,
  dependencies: [glib_dep, gst_dep, gst_base_dep, pahomqttc_dep],
  install: false,
  install_dir: nnstreamer_libdir
)

gstmqtt_lib = gstmqtt_shared
if get_option('default_library') == 'static'
  gstmqtt_lib = gstmqtt_static
endif

gstmqtt_dep = declare_dependency(link_with: gstmqtt_lib)
