tizen_apptest_deps = [
  capi_base_common_dep,
  dlog_dep,
  tizen_capi_dep,
  gtest_dep
]

unittest_tizen_capi = executable('unittest_tizen_capi',
  'unittest_tizen_capi.cpp',
  dependencies: [tizen_apptest_deps],
  install: false
)
test('unittest_tizen_capi', unittest_tizen_capi, args: ['--gst-plugin-path=../..'])
