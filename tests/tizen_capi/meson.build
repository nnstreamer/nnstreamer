tizen_apptest_deps = [
  nnstreamer_capi_dep,
  gtest_dep,
  glib_dep
]

unittest_tizen_capi = executable('unittest_tizen_capi',
  'unittest_tizen_capi.cpp',
  dependencies: [tizen_apptest_deps],
  install: false
)
test('unittest_tizen_capi', unittest_tizen_capi, args: ['--gst-plugin-path=../..'])

tizen_apptest_deps = [
  nnstreamer_capi_single_new_dep,
  gtest_dep,
  glib_dep
]

unittest_tizen_capi_single_new = executable('unittest_tizen_capi_single_new',
  'unittest_tizen_capi.cpp',
  dependencies: [tizen_apptest_deps],
  install: false
)
test('unittest_tizen_capi_single_new', unittest_tizen_capi_single_new, args: ['--gst-plugin-path=../..'])
