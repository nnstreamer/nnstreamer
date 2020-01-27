unittest_mvncsdk_helper = shared_library('mvnc',
  ['NCSDKTensorFilterTestHelper.cc'],
  dependencies : [glib_dep, gst_base_dep, gst_dep]
)

unittest_mvncsdk_helper_dep = declare_dependency(
  link_with: unittest_mvncsdk_helper,
)

unittest_mvncsdk_deps = [
  gtest_dep,
  glib_dep,
  gst_dep,
  gst_base_dep,
  nnstreamer_dep, # need to modify this to get the dummy loaded
  unittest_mvncsdk_helper_dep,
]

unittest_mvncsdk = executable('unittest_filter_mvncsdk2',
    ['unittest_filter_mvncsdk2.cc'],
    dependencies: unittest_mvncsdk_deps,
    install: get_option('install-test'),
    install_dir: unittest_install_dir
  )

test('LD_PRELOAD=./libmvnc.so unittest_filter_mvncsdk2', unittest_mvncsdk, args: ['--gst-plugin-path=../../'])
