if nnstreamer_edge_support_is_available
  subdir('edge')
endif
subdir('join')
if mqtt_support_is_available
  subdir('mqtt')
endif
subdir('nnstreamer')

if get_option('enable-datarepo')
  json_glib_dep = dependency('json-glib-1.0', required : true, \
      not_found_message : 'json-glib-1.0 package should be installed to build the datarepo elements.')
  subdir('datarepo')
endif
