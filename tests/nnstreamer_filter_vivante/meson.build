# The Vivante SDK is proprietary and is not available to the CI machines, so
# the sub-plugin is built again here against a mock of ovxlib. This does not
# touch the build of the sub-plugin itself, which stays behind 'enable-vivante'.
vivante_mock_incs = include_directories('.')

vivante_subplugin_src = join_paths(meson.project_source_root(), 'ext',
  'nnstreamer', 'tensor_filter', 'vivante', 'tensor_filter_vivante.c')

mock_ovxlib_lib = shared_library('mock_ovxlib',
  ['mock_ovxlib.c'],
  include_directories: vivante_mock_incs,
  dependencies: [glib_dep],
  install: false
)

mock_ovxlib_dep = declare_dependency(
  link_with: mock_ovxlib_lib,
  include_directories: vivante_mock_incs,
)

mock_vivante_model_lib = shared_library('mock_vivante_model',
  ['mock_vivante_model.c'],
  c_args: '-DNNS_MOCK_MODEL_FULL',
  dependencies: [glib_dep, mock_ovxlib_dep],
  install: false
)

# The same model library under another name, to exercise reopening a filter.
mock_vivante_model_alt_lib = shared_library('mock_vivante_model_alt',
  ['mock_vivante_model.c'],
  c_args: '-DNNS_MOCK_MODEL_FULL',
  dependencies: [glib_dep, mock_ovxlib_dep],
  install: false
)

# A model library missing the symbols that the sub-plugin resolves last.
mock_vivante_model_partial_lib = shared_library('mock_vivante_model_partial',
  ['mock_vivante_model.c'],
  dependencies: [glib_dep, mock_ovxlib_dep],
  install: false
)

# The paths of the mock model libraries are baked in as build-directory
# absolute paths, so neither this binary nor those libraries are installed;
# the meson test suite is what runs them.
unittest_filter_vivante = executable('unittest_filter_vivante',
  ['unittest_filter_vivante.cc', vivante_subplugin_src],
  cpp_args: [
    '-DMOCK_VIVANTE_MODEL_PATH="' + mock_vivante_model_lib.full_path() + '"',
    '-DMOCK_VIVANTE_MODEL_ALT_PATH="' + mock_vivante_model_alt_lib.full_path() + '"',
    '-DMOCK_VIVANTE_MODEL_PARTIAL_PATH="' + mock_vivante_model_partial_lib.full_path() + '"'
  ],
  link_depends: [mock_vivante_model_lib, mock_vivante_model_alt_lib,
    mock_vivante_model_partial_lib],
  dependencies: [nnstreamer_unittest_deps, nnstreamer_single_dep,
    mock_ovxlib_dep, libdl_dep],
  install: false
)

test('unittest_filter_vivante', unittest_filter_vivante, env: testenv)
