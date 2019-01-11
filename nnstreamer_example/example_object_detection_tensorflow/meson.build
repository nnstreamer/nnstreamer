cairo_dep = dependency('cairo')

nnstreamer_example_object_detection_tensorflow = executable('nnstreamer_example_object_detection_tensorflow',
  'nnstreamer_example_object_detection_tensorflow.cc',
  dependencies: [glib_dep, gst_dep, gst_video_dep, cairo_dep, libm_dep],
  include_directories: nnstreamer_inc,
  install: get_option('install-example'),
  install_dir: examples_install_dir
)

if get_option('install-example')
  install_data(['get_model.sh'],
    install_dir: examples_install_dir
  )
endif
