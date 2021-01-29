test_nnfw_deps = [glib_dep, gst_dep, nnstreamer_dep, gtest_dep, nnfw_plugin_dep, unittest_util_dep]

if get_option('enable-capi')
  unittest_nnfw_runtime_api = executable('unittest_nnfw_runtime_api',
    'unittest_tizen_nnfw_runtime_api.cc',
    dependencies: [test_nnfw_deps, nnstreamer_capi_dep],
    install: get_option('install-test'),
    install_dir: join_paths(unittest_install_dir,'tests'),
  )

  test('unittest_nnfw_runtime_api', unittest_nnfw_runtime_api, env: testenv)
endif

unittest_nnfw_runtime_raw = executable('unittest_nnfw_runtime_raw',
  'unittest_tizen_nnfw_runtime_raw.cc',
  dependencies: test_nnfw_deps,
  install: get_option('install-test'),
  install_dir: join_paths(unittest_install_dir,'tests'),
)

test('unittest_nnfw_runtime_raw', unittest_nnfw_runtime_raw, env: testenv)
