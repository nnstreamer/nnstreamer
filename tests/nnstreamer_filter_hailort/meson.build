# HailoRT and the Hailo NPUs are not available to the CI machines, so the
# sub-plugin is built again here against a mock of the HailoRT C API. This
# does not touch the build of the sub-plugin itself, which stays behind
# 'hailort-support'. Nothing here is installed: the SSAT cases and the unit
# tests find these libraries in the build directory.
hailort_mock_incs = include_directories('mock')

hailort_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'tensor_filter_hailort.cc')

hailort_mock_lib = shared_library('hailort_mock',
  [join_paths('mock', 'hailort_mock.cc')],
  include_directories: hailort_mock_incs,
  dependencies: [glib_dep],
  install: false
)

hailort_mock_dep = declare_dependency(
  link_with: hailort_mock_lib,
  include_directories: hailort_mock_incs,
)

# The sub-plugin on the mock, for runTest.sh in this directory.
shared_library('nnstreamer_filter_hailort',
  [hailort_subplugin_src],
  dependencies: [glib_dep, nnstreamer_single_dep, hailort_mock_dep],
  install: false
)
