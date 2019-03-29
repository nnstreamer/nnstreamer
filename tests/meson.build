# These custom filters are used in unittest_sink.
library('nnscustom_framecounter',
  join_paths('nnstreamer_sink', 'nnscustom_framecounter.c'),
  dependencies: [glib_dep, nnstreamer_dep],
  install: install_test,
  install_dir: customfilter_install_dir
)

library('nnscustom_drop_buffer',
  join_paths('nnstreamer_sink', 'nnscustom_drop_buffer.c'),
  dependencies: [nnstreamer_dep],
  install: install_test,
  install_dir: customfilter_install_dir
)

# Build and copy exe for ssat
libpng_dep = dependency('libpng', required: false)
if libpng_dep.found()
  b2p = executable('bmp2png',
    'bmp2png.c',
    dependencies: [libpng_dep],
    install: install_test,
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
  install: install_test,
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
    install: install_test,
    install_dir: unittest_install_dir
  )

  test('unittest_common', unittest_common)

  # Run unittest_sink
  unittest_sink = executable('unittest_sink',
    join_paths('nnstreamer_sink', 'unittest_sink.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: install_test,
    install_dir: unittest_install_dir
  )

  test('unittest_sink', unittest_sink, args: ['--gst-plugin-path=..'])

  # Run unittest_plugins
  unittest_plugins = executable('unittest_plugins',
    join_paths('nnstreamer_plugins', 'unittest_plugins.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: install_test,
    install_dir: unittest_install_dir
  )

  test('unittest_plugins', unittest_plugins, args: ['--gst-plugin-path=..'])

  # Run unittest_src_iio
  unittest_src_iio = executable('unittest_src_iio',
    join_paths('nnstreamer_source', 'unittest_src_iio.cpp'),
    dependencies: [nnstreamer_unittest_deps],
    install: install_test,
    install_dir: unittest_install_dir
  )

  test('unittest_src_iio', unittest_src_iio, args: ['--gst-plugin-path=..'])
endif

# Tizen C-API
if get_option('enable-tizen-capi')
  subdir('tizen_capi')
endif
