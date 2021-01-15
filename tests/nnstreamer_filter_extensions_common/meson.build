if not gtest_dep.found()
    error('Cannot find gtest library')
endif

tizen_apptest_deps = [
  gtest_dep,
  glib_dep
]

# Format for adding subplugin into extensions -
# [name, extension abbreviation, dependencies, model file name/folder path/file path, test name]
extensions = []
custom_filter_path = join_paths(meson.build_root(), 'nnstreamer_example',
    'libnnstreamer_customfilter_passthrough.' + so_ext)
extensions += [['custom', 'custom', nnstreamer_unittest_deps, custom_filter_path, 'custom']]
extensions += [['custom', 'custom', nnstreamer_unittest_deps, custom_filter_path, 'custom-set']]

if tflite_support_is_available
  extensions += [['tensorflow-lite', 'tflite', nnstreamer_filter_tflite_deps, 'mobilenet_v1_1.0_224_quant.tflite', 'tensorflow_lite']]
  extensions += [['tensorflow-lite', 'tflite', nnstreamer_filter_tflite_deps, 'add.tflite', 'tensorflow_lite-set']]
endif

if tf_support_is_available
  extensions += [['tensorflow', 'tf', nnstreamer_filter_tf_deps, 'mnist.pb', 'tensorflow']]
endif

if nnfw_runtime_support_is_available
  extensions += [['nnfw', 'nnfw', nnstreamer_filter_nnfw_deps, 'add.tflite', 'nnfw']]
endif

if pytorch_support_is_available
  extensions += [['pytorch', 'torch', nnstreamer_filter_torch_deps, 'pytorch_lenet5.pt', 'pytorch']]
endif

if caffe2_support_is_available
  extensions += [['caffe2', 'caffe2', nnstreamer_filter_caffe2_deps, 'caffe2_init_net.pb,caffe2_predict_net.pb', 'caffe2']]
endif

if have_python3
  extensions += [['python3', 'python3', nnstreamer_filter_python3_deps, 'passthrough.py', 'python3-get']]
  extensions += [['python3', 'python3', nnstreamer_filter_python3_deps, 'scaler.py', 'python3-set']]
endif

sed_command = find_program('sed', required: true)
ext_test_template_prefix = 'unittest_tizen_'
ext_test_template = files (ext_test_template_prefix + 'template.cc.in')

foreach ext : extensions
  ext_test_path_each = ext_test_template_prefix + ext[4] + '.cc'

  sed_ext_name_option = 's|@EXT_NAME@|' + ext[0] + '|'
  sed_ext_abbrv_option = 's|@EXT_ABBRV@|' + ext[1] + '|'
  sed_ext_mf_option = 's|@MODEL_FILE@|' + ext[3] + '|'

  ext_test_each = custom_target (
    ext_test_path_each,
    input : ext_test_template,
    output : ext_test_path_each,
    command : [copy, '-f', '@INPUT@', '@OUTPUT@', \
      '&&', 'sed', '-i', sed_ext_name_option, '@OUTPUT@', \
      '&&', 'sed', '-i', sed_ext_abbrv_option, '@OUTPUT@', \
      '&&', 'sed', '-i', sed_ext_mf_option, '@OUTPUT@']
  )

  exec = executable(
    ext_test_template_prefix + ext[4],
    ext_test_each,
    dependencies: [tizen_apptest_deps, ext[2]],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  filter_ext_common_testenv = environment()
  filter_ext_common_testenv.set('GST_PLUGIN_PATH', path_gst_plugin)
  filter_ext_common_testenv.set('NNSTREAMER_CONF', path_nns_conf)
  filter_ext_common_testenv.set('NNSTREAMER_FILTERS', path_nns_plugin_filters)
  filter_ext_common_testenv.set('NNSTREAMER_DECODERS', path_nns_plugin_decoders)
  filter_ext_common_testenv.set('NNSTREAMER_CONVERTERS', path_nns_plugin_converters)
  filter_ext_common_testenv.set('NNSTREAMER_SOURCE_ROOT_PATH', meson.source_root())

  if ext[0] == 'python2'
    py2_module_path = join_paths(meson.build_root(), 'tests/' + ext[0] + '_module')
    run_command('rm', '-rf', py2_module_path, check : true)
    run_command('mkdir', '-p', py2_module_path, check : true)
    run_command('ln', '-sf',
        join_paths(meson.build_root(), 'ext/nnstreamer/tensor_filter/nnstreamer_python2.' + so_ext),
        py2_module_path + '/nnstreamer_python.so', check : true)
    filter_ext_common_testenv.set('PYTHONPATH', py2_module_path)
  elif ext[0] == 'python3'
    py3_module_path = join_paths(meson.build_root(), 'tests/' + ext[0] + '_module')
    run_command('rm', '-rf', py3_module_path, check : true)
    run_command('mkdir', '-p', py3_module_path, check : true)
    run_command('ln', '-sf',
        join_paths(meson.build_root(), 'ext/nnstreamer/tensor_filter/nnstreamer_python3.' + so_ext),
        py3_module_path + '/nnstreamer_python.so', check : true)
    filter_ext_common_testenv.set('PYTHONPATH', py3_module_path)
  endif

  test(ext_test_template_prefix + ext[4], exec, env: filter_ext_common_testenv)
endforeach
