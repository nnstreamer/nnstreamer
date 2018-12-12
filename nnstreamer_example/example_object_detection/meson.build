cairo_dep = dependency('cairo')

nnstreamer_example_object_detection = executable('nnstreamer_example_object_detection',
  'nnstreamer_example_object_detection.cc',
  dependencies: [glib_dep, gst_dep, gst_video_dep, cairo_dep, libm_dep],
  include_directories: nnstreamer_inc,
  install: get_option('INSTALL_EXAMPLES'),
  install_dir: examples_install_dir
)

if get_option('INSTALL_EXAMPLES')
  install_data(['get_model.sh', 'gst-launch-object-detection.sh'],
    install_dir: examples_install_dir
  )
endif
