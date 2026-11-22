# The QNN SDK is not available to the CI machines, so the qnn sub-plugin is
# built again for unittest_filter_qnn (tests/meson.build) against the reduced
# QNN header in mock/, and opens these mock model and backend libraries
# instead of a QNN model and backend. This does not touch the build of the
# sub-plugin itself, which stays behind 'qnn-support'. Nothing here is
# installed.
qnn_mock_incs = include_directories('mock')

qnn_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'tensor_filter_qnn.cc')

qnn_mock_backend_lib = shared_library('qnn_mock_backend',
  [join_paths('mock', 'qnn_mock_backend.cc')],
  include_directories: qnn_mock_incs,
  install: false
)

qnn_mock_model_lib = shared_library('qnn_mock_model',
  [join_paths('mock', 'qnn_mock_model.cc')],
  include_directories: qnn_mock_incs,
  install: false
)

qnn_mock_dep = declare_dependency(
  link_with: [qnn_mock_backend_lib, qnn_mock_model_lib],
  include_directories: qnn_mock_incs,
  compile_args: [
    '-DQNN_MOCK_BACKEND_PATH="' + qnn_mock_backend_lib.full_path() + '"',
    '-DQNN_MOCK_MODEL_PATH="' + qnn_mock_model_lib.full_path() + '"',
  ],
)
