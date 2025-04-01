# Test lib for nnstreamer-edge custom connection
library('nnstreamer-edge-custom-test',
  join_paths('nnstreamer-edge-custom-test.c'),
  dependencies: edge_deps,
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

unittest_edge = executable('unittest_edge',
  join_paths('edge', 'unittest_edge.cc'),
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

# Let the test lib for custom connection be loaded runtime.
testenv.append('LD_LIBRARY_PATH', meson.current_build_dir())

test('unittest_edge', unittest_edge, env: testenv)

# Run unittest_query
unittest_query = executable('unittest_query',
  join_paths('query', 'unittest_query.cc'),
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)

test('unittest_query', unittest_query, env: testenv)
