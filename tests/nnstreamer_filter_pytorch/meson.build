# LibTorch is too large to install in every CI job, so the sub-plugin is built
# again here against a reduced stand-in of its API. The build of the sub-plugin
# itself stays behind 'pytorch-support'; .github/workflows/pytorch.yml checks it
# against a real LibTorch every week.
pytorch_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'tensor_filter_pytorch.cc')

unittest_filter_pytorch_mock = executable('unittest_filter_pytorch_mock',
  ['unittest_filter_pytorch_mock.cc', join_paths('mock', 'torch_mock.cc'),
    pytorch_subplugin_src],
  include_directories: include_directories('mock'),
  dependencies: [nnstreamer_unittest_deps, nnstreamer_single_dep],
  install: false
)

test('unittest_filter_pytorch_mock', unittest_filter_pytorch_mock, env: testenv)
# nnstreamer caches a configuration value at its first lookup, so the other
# choice of the configuration needs a run of its own.
test('unittest_filter_pytorch_mock_gpu_conf', unittest_filter_pytorch_mock,
  args: ['--gtest_filter=*.openDeviceByConfiguration', '--gpu-by-conf'],
  env: testenv)
