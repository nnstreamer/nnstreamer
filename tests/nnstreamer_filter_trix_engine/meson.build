# npu-engine, the TRIx NPU runtime, is not available to the CI machines, so
# the trix-engine sub-plugin is built again for unittest_filter_trix_engine
# (tests/meson.build) against the mock npu-engine in mock/. This does not
# touch the build of the sub-plugin itself, which stays behind
# 'trix-engine-support'. Nothing here is installed.
trix_mock_incs = include_directories('mock',
  join_paths('..', '..', 'ext', 'nnstreamer', 'tensor_filter'))

trix_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'tensor_filter_trix_engine.cc')

trix_mock_lib = shared_library('trix_mock',
  [join_paths('mock', 'trix_mock.cc')],
  include_directories: trix_mock_incs,
  install: false
)

trix_mock_dep = declare_dependency(
  link_with: trix_mock_lib,
  include_directories: trix_mock_incs,
)
