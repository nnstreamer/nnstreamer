subdir('nnstreamer_tensors/tensors_test')

libpng_dep = dependency('libpng', required: false)
if libpng_dep.found()
  executable('bmp2png',
    'bmp2png.c',
    dependencies: [libpng_dep]
  )
endif

gtest_dep = dependency('gtest', required: false)
if gtest_dep.found()
  # This custom filter is used in unittest_sink.
  nnscustom_framecounter_lib = library('nnscustom_framecounter',
    'nnstreamer_sink/nnscustom_framecounter.c',
    c_args: nnstreamer_base_args,
    dependencies: [glib_dep],
    include_directories: nnstreamer_inc)

  nnstreamer_unittest_deps = [
    nnstreamer_base_deps,
    nnstreamer_dep,
    gst_app_dep,
    gst_check_dep,
    gtest_dep
  ]

  # Run unittest_common
  unittest_common = executable('unittest_common',
    'common/unittest_common.cpp',
    cpp_args: nnstreamer_base_args,
    dependencies: [nnstreamer_unittest_deps])

  test('unittest_common', unittest_common)

  # Run unittest_sink
  unittest_sink = executable('unittest_sink',
    'nnstreamer_sink/unittest_sink.cpp',
    cpp_args: nnstreamer_base_args,
    dependencies: [nnstreamer_unittest_deps])

  test('unittest_sink', unittest_sink, args: ['--gst-plugin-path=../'])

  # Run unittest_plugins
  unittest_plugins = executable('unittest_plugins',
    'nnstreamer_plugins/unittest_plugins.cpp',
    cpp_args: nnstreamer_base_args,
    dependencies: [nnstreamer_unittest_deps])

  test('unittest_plugins', unittest_plugins, args: ['--gst-plugin-path=../'])
endif
