tizen_apptest_deps = [
  gtest_dep,
  glib_dep
]

# TODO: enable for python

# Format for adding subplugin into extensions -
# [name, nick name, dependencies, model file name/folder path/file path]
extensions = []

extensions += [['custom', 'custom', nnstreamer_unittest_deps, '../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough.so']]

if get_option('enable-tensorflow-lite')
  extensions += [['tensorflow-lite', 'tflite', nnstreamer_filter_tflite_deps, 'mobilenet_v1_1.0_224_quant.tflite']]
endif

if get_option('enable-tensorflow')
  extensions += [['tensorflow', 'tf', nnstreamer_filter_tf_deps, 'mnist.pb']]
endif

if get_option('enable-nnfw-runtime')
  extensions += [['nnfw', 'nnfw', nnstreamer_filter_nnfw_deps, 'add.tflite']]
endif

if get_option('enable-pytorch')
  extensions += [['pytorch', 'torch', nnstreamer_filter_torch_deps, 'pytorch_lenet5.pt']]
endif

if get_option('enable-caffe2')
  extensions += [['caffe2', 'caffe2', nnstreamer_filter_caffe2_deps, 'caffe2_init_net.pb,caffe2_predict_net.pb']]
endif

if have_python2
  extensions += [['python2', 'python2', nnstreamer_filter_python2_deps, 'passthrough.py']]
endif

if have_python3
  extensions += [['python3', 'python3', nnstreamer_filter_python3_deps, 'passthrough.py']]
endif

foreach ext : extensions
  gen_test_src = run_command('deploy.sh', ext[0], ext[1], ext[3])

  if gen_test_src.returncode() == 0
    exec = executable('unittest_tizen_' + ext[0],
      ['unittest_tizen_' + ext[0] + '.cpp'],
      dependencies: [tizen_apptest_deps, ext[2]],
      install: get_option('install-test'),
      install_dir: unittest_install_dir
    )

    test('unittest_tizen_' + ext[0], exec, args: ['--gst-plugin-path=../..'])
  else
    warning('Failed to generate basic unittest for ' + ext[0])
  endif
endforeach
