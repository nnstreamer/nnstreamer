mqtt_plugin_srcs = [
  join_paths(meson.current_source_dir(), 'mqttsink.c'),
  join_paths(meson.current_source_dir(), 'mqttsrc.c'),
  join_paths(meson.current_source_dir(), 'mqttelements.c'),
]

gstmqtt_shared = shared_library('gstmqtt',
  mqtt_plugin_srcs,
  dependencies: [glib_dep, gst_dep, gst_base_dep, pahomqttc_dep],
  install: true,
  install_dir: plugins_install_dir
)

# gstmqtt_shared is a GStreamer plugin so that providing runtime dependency is not necessary.
# This dependency is needed to let the class that mocks the MQTT library use the internal header files.
gstmqtt_mocking_dep = declare_dependency(
  include_directories: '.',
)
