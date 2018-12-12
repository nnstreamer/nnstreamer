cairo_dep = dependency('cairo')

nnstreamer_example_object_detection = executable('nnstreamer_example_object_detection',
  'nnstreamer_example_object_detection.cc',
  dependencies: [glib_dep, gst_dep, gst_video_dep, cairo_dep, libm_dep],
  include_directories: nnstreamer_inc
)
