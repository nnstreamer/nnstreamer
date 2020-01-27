unittest_mvncsdk_helper = shared_library('mvnc',
  ['NCSDKTensorFilterTestHelper.cc'],
  dependencies : [glib_dep, gst_base_dep, gst_dep]
)

unittest_mvncsdk_helper_dep = declare_dependency(
  link_with: unittest_mvncsdk_helper,
)
