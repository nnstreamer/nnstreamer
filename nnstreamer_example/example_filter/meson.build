nnstreamer_example_filter = executable('nnstreamer_example_filter',
  'nnstreamer_example_filter.c',
  dependencies: [glib_dep, gst_dep],
  install: get_option('install-example'),
  install_dir: examples_install_dir
)

if get_option('install-example')
  install_data('nnstreamer_example_filter.py',
    install_dir: examples_install_dir
  )
endif
