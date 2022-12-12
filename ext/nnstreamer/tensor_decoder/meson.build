# direct video
decoder_sub_direct_video_sources = [
  'tensordec-directvideo.c',
  'tensordecutil.c'
]

shared_library('nnstreamer_decoder_direct_video',
  decoder_sub_direct_video_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_direct_video',
  decoder_sub_direct_video_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# image labeling
decoder_sub_image_labeling_sources = [
  'tensordec-imagelabel.c',
  'tensordecutil.c'
]

shared_library('nnstreamer_decoder_image_labeling',
  decoder_sub_image_labeling_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_image_labeling',
  decoder_sub_image_labeling_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# bounding boxes
decoder_sub_bounding_boxes_sources = [
  'tensordec-boundingbox.c',
  'tensordecutil.c',
  'tensordec-font.c'
]

shared_library('nnstreamer_decoder_bounding_boxes',
  decoder_sub_bounding_boxes_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep, libm_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_bounding_boxes',
  decoder_sub_bounding_boxes_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep, libm_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# pose estimation
decoder_sub_pose_estimation_sources = [
  'tensordec-pose.c',
  'tensordecutil.c',
  'tensordec-font.c'
]

shared_library('nnstreamer_decoder_pose_estimation',
  decoder_sub_pose_estimation_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_pose_estimation',
  decoder_sub_pose_estimation_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# image segmentation
decoder_sub_image_segment_sources = [
  'tensordec-imagesegment.c',
  'tensordecutil.c'
]

shared_library('nnstreamer_decoder_image_segment',
  decoder_sub_image_segment_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_image_segment',
  decoder_sub_image_segment_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# protobuf decoder
if protobuf_support_is_available
  decoder_sub_protobuf_sources = [
    'tensordec-protobuf.cc',
    'tensordecutil.c'
  ]

  shared_library('nnstreamer_decoder_protobuf',
    decoder_sub_protobuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, protobuf_util_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir
  )
endif

# flatbuffer
if flatbuf_support_is_available
  decoder_sub_flatbuf_sources = [
    'tensordec-flatbuf.cc',
    'tensordecutil.c'
  ]

  decoder_sub_flatbuf_sources += fb_gen_src

  shared_library('nnstreamer_decoder_flatbuf',
    decoder_sub_flatbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir,
  )

  decoder_sub_flexbuf_sources = [
    'tensordec-flexbuf.cc',
    'tensordecutil.c'
  ]

  shared_library('nnstreamer_decoder_flexbuf',
    decoder_sub_flexbuf_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir,
  )
endif

# python3
if have_python3
  decoder_sub_python3_sources = ['tensordec-python3.cc', 'tensordecutil.c']

  shared_library('nnstreamer_decoder_python3',
    decoder_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: decoder_subplugin_install_dir
  )

  static_library('nnstreamer_decoder_python3',
    decoder_sub_python3_sources,
    dependencies: nnstreamer_python3_helper_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

# octet stream
decoder_sub_octet_stream_sources = [
  'tensordec-octetstream.c',
  'tensordecutil.c'
]

shared_library('nnstreamer_decoder_octet_stream',
  decoder_sub_octet_stream_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_octet_stream',
  decoder_sub_octet_stream_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)
