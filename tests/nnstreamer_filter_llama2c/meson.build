# The llama2c library (https://github.com/nnsuite/llama2.c) is not installed
# on the CI machines, so the llama2c sub-plugin is built again for
# unittest_filter_llama2c (tests/meson.build) against the mock llama2c in
# mock/. This does not touch the build of the sub-plugin itself, which stays
# behind 'llama2c-support'. Nothing here is installed.
llama2c_mock_incs = include_directories('mock')

llama2c_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'tensor_filter_llama2.cc')

llama2c_mock_lib = shared_library('llama2c_mock',
  [join_paths('mock', 'llama2c_mock.cc')],
  include_directories: llama2c_mock_incs,
  install: false
)

llama2c_mock_dep = declare_dependency(
  link_with: llama2c_mock_lib,
  include_directories: llama2c_mock_incs,
)
