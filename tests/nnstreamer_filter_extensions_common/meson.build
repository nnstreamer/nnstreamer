if not gtest_dep.found()
    error('Cannot find gtest library')
endif

tizen_apptest_deps = [
  gtest_dep,
  glib_dep
]

# Format for adding subplugin into extensions -
# {execute_when, name, ext_abbr, test_name, deps, model file/folder, env}

py_env = environment()

py_env.append('PYTHONPATH', join_paths(meson.build_root(), 'ext', 'nnstreamer', 'tensor_filter'))

nnstreamer_example_dir = join_paths(meson.build_root(), 'nnstreamer_example')

extensions = [
  {
    'execute_when': true,
    'name': 'custom',
    'abbr': 'custom',
    'test_name': 'custom',
    'deps': 'nnstreamer_unittest_deps',
    'model': join_paths(nnstreamer_example_dir, 'custom_example_passthrough', 'libnnstreamer_customfilter_passthrough.so')
  },
  {
    'execute_when': true,
    'name': 'custom',
    'abbr': 'custom',
    'test_name': 'custom-set',
    'deps': 'nnstreamer_unittest_deps',
    'model': join_paths(nnstreamer_example_dir, 'custom_example_passthrough', 'libnnstreamer_customfilter_passthrough_variable.so')
  },
  {
    'execute_when': get_option('enable-tensorflow-lite'),
    'name': 'tensorflow-lite',
    'abbr': 'tflite',
    'test_name': 'tensorflow_lite',
    'deps': 'nnstreamer_filter_tflite_deps',
    'model': 'mobilenet_v1_1.0_224_quant.tflite'
  },
  {
    'execute_when': get_option('enable-tensorflow-lite'),
    'name': 'tensorflow-lite',
    'abbr': 'tflite',
    'test_name': 'tensorflow_lite-set',
    'deps': 'nnstreamer_filter_tflite_deps',
    'model': 'add.tflite'
  },
  {
    'execute_when': get_option('enable-tensorflow'),
    'name': 'tensorflow',
    'abbr': 'tf',
    'test_name': 'tensorflow',
    'deps': 'nnstreamer_filter_tf_deps',
    'model': 'mnist.pb'
  },
  {
    'execute_when': get_option('enable-nnfw-runtime'),
    'name': 'nnfw',
    'abbr': 'nnfw',
    'test_name': 'nnfw',
    'deps': 'nnstreamer_filter_nnfw_deps',
    'model': 'add.tflite'
  },
  {
    'execute_when': get_option('enable-pytorch'),
    'name': 'pytorch',
    'abbr': 'torch',
    'test_name': 'pytorch',
    'deps': 'nnstreamer_filter_torch_deps',
    'model': 'pytorch_lenet5.pt'
  },
  {
    'execute_when': get_option('enable-caffe2'),
    'name': 'caffe2',
    'abbr': 'caffe2',
    'test_name': 'caffe2',
    'deps': 'nnstreamer_filter_caffe2_deps',
    'model': 'caffe2_init_net.pb,caffe2_predict_net.pb'
  },
  {
    'execute_when': have_python2,
    'name': 'python2',
    'abbr': 'python2',
    'test_name': 'python2-get',
    'deps': 'nnstreamer_filter_python2_deps',
    'model': 'passthrough.py',
    'env': py_env
  },
  {
    'execute_when': have_python2,
    'name': 'python2',
    'abbr': 'python2',
    'test_name': 'python2-set',
    'deps': 'nnstreamer_filter_python2_deps',
    'model': 'scaler.py',
    'env': py_env
  },  
  {
    'execute_when': have_python3,
    'name': 'python3',
    'abbr': 'python3',
    'test_name': 'python3-get',
    'deps': 'nnstreamer_filter_python3_deps',
    'model': 'passthrough.py',
    'env': py_env
  },
  {
    'execute_when': have_python3,
    'name': 'python3',
    'abbr': 'python3',
    'test_name': 'python3-set',
    'deps': 'nnstreamer_filter_python3_deps',
    'model': 'scaler.py',
    'env': py_env
  },
]

sed_command = find_program('sed', required: true)
ext_test_template_prefix = 'unittest_tizen_'
ext_test_template = files (ext_test_template_prefix + 'template.cc.in')

foreach ext : extensions
  if not ext.get('execute_when', false)
    continue
  endif

  ext_test_path_each = ext_test_template_prefix + ext['test_name'] + '.cc'

  sed_ext_name_option = 's|@EXT_NAME@|' + ext['name'] + '|'
  sed_ext_abbrv_option = 's|@EXT_ABBRV@|' + ext['abbr'] + '|'
  sed_ext_mf_option = 's|@MODEL_FILE@|' + ext['model'] + '|'

  ext_test_each = custom_target (
    ext_test_path_each,
    input : ext_test_template,
    output : ext_test_path_each,
    command : [copy, '-f', '@INPUT@', '@OUTPUT@', \
      '&&', 'sed', '-i\'.cc\'', sed_ext_name_option, '@OUTPUT@', \
      '&&', 'sed', '-i\'.cc\'', sed_ext_abbrv_option, '@OUTPUT@', \
      '&&', 'sed', '-i\'.cc\'', sed_ext_mf_option, '@OUTPUT@']
  )

  _deps = get_variable(ext['deps'], [false])

  if false in _deps
    error('dependencies not found for @0@'.format(ext['test_name']))
  endif

  exec = executable(
    ext_test_template_prefix + ext['test_name'],
    ext_test_each,
    dependencies: [tizen_apptest_deps, _deps],
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

  test(ext_test_template_prefix + ext['test_name'], exec, args: ['--gst-plugin-path=../..'], env: ext.get('env', environment()))
endforeach
