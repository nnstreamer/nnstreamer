# custom_example_passthrough
library('nnstreamer_customfilter_passthrough',
  join_paths('custom_example_passthrough',
      'nnstreamer_customfilter_example_passthrough.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

library('nnstreamer_customfilter_passthrough_variable',
  join_paths('custom_example_passthrough',
      'nnstreamer_customfilter_example_passthrough_variable.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# custom_example_average
library('nnstreamer_customfilter_average',
  join_paths('custom_example_average',
      'nnstreamer_customfilter_example_average.c'),
  include_directories: nnstreamer_inc,
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# custom_example_scaler
library('nnstreamer_customfilter_scaler',
  join_paths('custom_example_scaler',
      'nnstreamer_customfilter_example_scaler.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

library('nnstreamer_customfilter_scaler_allocator',
  join_paths('custom_example_scaler',
      'nnstreamer_customfilter_example_scaler_allocator.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# custom_example_opencv
opencv_dep = dependency('opencv', required: false)
if opencv_dep.found()
  library('nnstreamer_customfilter_opencv_scaler',
    join_paths('custom_example_opencv',
        'nnstreamer_customfilter_opencv_scaler.cc'),
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, opencv_dep],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )

  library('nnstreamer_customfilter_opencv_average',
    join_paths('custom_example_opencv',
        'nnstreamer_customfilter_opencv_average.cc'),
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, opencv_dep],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )
endif

# custom_example_RNN
library('dummyRNN',
  join_paths('custom_example_RNN', 'dummy_RNN.c'),
  include_directories: nnstreamer_inc,
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# custom_example_LSTM
library('dummyLSTM',
  join_paths('custom_example_LSTM', 'dummy_LSTM.c'),
  dependencies: [libm_dep, glib_dep],
  include_directories: nnstreamer_inc,
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# This custom filter is used in unittest_sink.
library('nnscustom_drop_buffer',
  join_paths('custom_example_drop_buffer', 'nnscustom_drop_buffer.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# This custom filter is used in unittest_sink.
library('nnscustom_framecounter',
  join_paths('custom_example_framecounter', 'nnscustom_framecounter.c'),
  dependencies: [glib_dep, gst_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

if tensorrt_support_is_available
  library('nnstreamer_customfilter_tensorrt_reshape',
    'nnstreamer_customfilter_example_tensorrt_reshape.cc',
    dependencies: [glib_dep, gst_dep, nnstreamer_dep, tensorrt_deps],
    install: get_option('install-test'),
    install_dir: customfilter_install_dir
  )
endif

