# These custom filters are used in unittest_sink.
library('nnscustom_framecounter',
  join_paths('nnstreamer_sink', 'nnscustom_framecounter.c'),
  dependencies: [glib_dep, nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

library('nnscustom_drop_buffer',
  join_paths('nnstreamer_sink', 'nnscustom_drop_buffer.c'),
  dependencies: [nnstreamer_dep],
  install: get_option('install-test'),
  install_dir: customfilter_install_dir
)

# Build and copy exe for ssat
libpng_dep = dependency('libpng', required: false)
if libpng_dep.found()
  b2p = executable('bmp2png',
    'bmp2png.c',
    dependencies: [libpng_dep],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  copy = find_program('cp')
  custom_target('copy-bmp2png',
    input: b2p,
    output: 'b2p',
    command: [copy, '@INPUT@', meson.current_source_dir()],
    build_by_default: true
  )
endif

# ssat repo_dynamic
executable('unittest_repo',
  join_paths('nnstreamer_repo_dynamicity', 'tensor_repo_dynamic_test.c'),
  dependencies: [nnstreamer_dep, gst_dep, glib_dep],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

# gtest
gtest_dep = dependency('gtest', required: false)
if gtest_dep.found()
  nnstreamer_unittest_deps = [
    nnstreamer_dep,
    nnstreamer_internal_deps,
    gst_app_dep,
    gst_check_dep,
    gtest_dep
  ]

  # Run unittest_common
  unittest_common = executable('unittest_common',
    join_paths('common', 'unittest_common.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_common', unittest_common)

  # Run unittest_sink
  unittest_sink = executable('unittest_sink',
    join_paths('nnstreamer_sink', 'unittest_sink.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_sink', unittest_sink, timeout: 120, args: ['--gst-plugin-path=..'])

  # Run unittest_plugins
  unittest_plugins = executable('unittest_plugins',
    join_paths('nnstreamer_plugins', 'unittest_plugins.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_plugins', unittest_plugins, args: ['--gst-plugin-path=..'])

  # Run unittest_src_iio
  unittest_src_iio = executable('unittest_src_iio',
    join_paths('nnstreamer_source', 'unittest_src_iio.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test('unittest_src_iio', unittest_src_iio, timeout: 120, args: ['--gst-plugin-path=..'])
endif

# Tizen C-API
if get_option('enable-tizen-capi')
  subdir('tizen_capi')
endif

# Install Unittest
if get_option('install-test')
  install_subdir('nnstreamer_converter', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_merge', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_decoder', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_demux', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_filter_custom', install_dir: join_paths(unittest_install_dir,'tests'))
  if get_option('enable-tensorflow-lite')
     install_subdir('nnstreamer_filter_tensorflow-lite', install_dir: join_paths(unittest_install_dir,'tests'))
     install_subdir('nnstreamer_decoder_image_labeling', install_dir: join_paths(unittest_install_dir,'tests'))
  endif
  if get_option('enable-tensorflow')
     install_subdir('nnstreamer_filter_tensorflow', install_dir: join_paths(unittest_install_dir,'tests'))
  endif
  install_subdir('nnstreamer_mux', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_repo', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_repo_dynamicity', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_repo_lstm', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_repo_rnn', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('nnstreamer_split', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('transform_arithmetic', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('transform_dimchg', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('transform_stand', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('transform_transpose', install_dir: join_paths(unittest_install_dir,'tests'))
  install_subdir('transform_typecast', install_dir: join_paths(unittest_install_dir,'tests'))
endif
