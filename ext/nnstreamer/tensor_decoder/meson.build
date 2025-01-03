# Common decoder util
decoder_util_sources = files('tensordecutil.c', 'tensordec-font.c')
decoder_util_deps = [nnstreamer_dep, glib_dep, gst_dep]

decoder_util_shared = shared_library('nnstreamer-decoder-util',
  decoder_util_sources,
  dependencies: decoder_util_deps,
  install: true,
  install_dir: nnstreamer_libdir
)
decoder_util_shared_dep = declare_dependency(
  link_with: decoder_util_shared,
  dependencies: decoder_util_deps
)

decoder_util_static = static_library('nnstreamer-decoder-util',
  decoder_util_sources,
  dependencies: decoder_util_deps,
  install: true,
  install_dir: nnstreamer_libdir
)
decoder_util_static_dep = declare_dependency(
  link_with: decoder_util_static,
  dependencies: decoder_util_deps
)

# direct video
decoder_sub_direct_video_sources = files('tensordec-directvideo.c')

shared_library('nnstreamer_decoder_direct_video',
  decoder_sub_direct_video_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_direct_video',
  decoder_sub_direct_video_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)

# image labeling
decoder_sub_image_labeling_sources = files('tensordec-imagelabel.c')

shared_library('nnstreamer_decoder_image_labeling',
  decoder_sub_image_labeling_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_image_labeling',
  decoder_sub_image_labeling_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)

# bounding boxes
decoder_sub_bounding_boxes_sources = files('tensordec-boundingbox.cc')

subdir('box_properties')
shared_library('nnstreamer_decoder_bounding_boxes',
  decoder_sub_bounding_boxes_sources,
  dependencies: [decoder_util_shared_dep, libm_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_bounding_boxes',
  decoder_sub_bounding_boxes_sources,
  dependencies: [decoder_util_static_dep, libm_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# tensor_region
decoder_sub_tensor_region_sources = files('tensordec-tensor_region.c')

shared_library('nnstreamer_decoder_tensor_region',
  decoder_sub_tensor_region_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)

static_library('nnstreamer_decoder_tensor_region',
  decoder_sub_tensor_region_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)

# pose estimation
decoder_sub_pose_estimation_sources = files('tensordec-pose.c')

shared_library('nnstreamer_decoder_pose_estimation',
  decoder_sub_pose_estimation_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_pose_estimation',
  decoder_sub_pose_estimation_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)

# image segmentation
decoder_sub_image_segment_sources = files('tensordec-imagesegment.c')

shared_library('nnstreamer_decoder_image_segment',
  decoder_sub_image_segment_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_image_segment',
  decoder_sub_image_segment_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)

# protobuf decoder
if protobuf_support_is_available
  decoder_sub_protobuf_sources = files('tensordec-protobuf.cc')

  shared_library('nnstreamer_decoder_protobuf',
    decoder_sub_protobuf_sources,
    dependencies: [decoder_util_shared_dep, protobuf_util_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir
  )
endif

# flatbuffer
if flatbuf_support_is_available
  decoder_sub_flatbuf_sources = files('tensordec-flatbuf.cc')

  shared_library('nnstreamer_decoder_flatbuf',
    decoder_sub_flatbuf_sources,
    dependencies: [decoder_util_shared_dep, flatbuf_util_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir,
  )

  decoder_sub_flexbuf_sources = files('tensordec-flexbuf.cc')

  shared_library('nnstreamer_decoder_flexbuf',
    decoder_sub_flexbuf_sources,
    dependencies: [decoder_util_shared_dep, flatbuf_util_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir,
  )
endif

# python3
if have_python3
  decoder_sub_python3_sources = files('tensordec-python3.cc')

  shared_library('nnstreamer_decoder_python3',
    decoder_sub_python3_sources,
    dependencies: [decoder_util_shared_dep, nnstreamer_python3_helper_dep],
    install: true,
    install_dir: decoder_subplugin_install_dir
  )

  static_library('nnstreamer_decoder_python3',
    decoder_sub_python3_sources,
    dependencies: [decoder_util_static_dep, nnstreamer_python3_helper_dep],
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

# octet stream
decoder_sub_octet_stream_sources = files('tensordec-octetstream.c')

shared_library('nnstreamer_decoder_octet_stream',
  decoder_sub_octet_stream_sources,
  dependencies: decoder_util_shared_dep,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_octet_stream',
  decoder_sub_octet_stream_sources,
  dependencies: decoder_util_static_dep,
  install: true,
  install_dir: nnstreamer_libdir
)
