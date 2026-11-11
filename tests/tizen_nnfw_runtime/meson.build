test_nnfw_deps = [nnstreamer_unittest_deps, nnfw_plugin_dep, libdl_dep]

unittest_nnfw_runtime_raw = executable('unittest_nnfw_runtime_raw',
  'unittest_tizen_nnfw_runtime_raw.cc',
  dependencies: test_nnfw_deps,
  # The test overrides nnfw_run () for the sub-plugin.
  export_dynamic: true,
  install: get_option('install-test'),
  install_dir: unittest_install_dir,
)

test('unittest_nnfw_runtime_raw', unittest_nnfw_runtime_raw, env: testenv)
