tensor_decoder_directvideo_shared = shared_library('nnstreamer_decoder_direct_video',
    [ 'tensordec-directvideo.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
tensor_decoder_directvideo_static = static_library('nnstreamer_decoder_direct_video',
    [ 'tensordec-directvideo.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

tensor_decoder_imagelabel_shared = shared_library('nnstreamer_decoder_image_labeling',
    [ 'tensordec-imagelabel.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
tensor_decoder_imagelabel_static = static_library('nnstreamer_decoder_image_labeling',
    [ 'tensordec-imagelabel.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)

tensor_decoder_boundingbox_shared = shared_library('nnstreamer_decoder_bounding_boxes',
    [ 'tensordec-boundingbox.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: decoder_subplugin_install_dir
)
tensor_decoder_boundingbox_static = static_library('nnstreamer_decoder_bounding_boxes',
    [ 'tensordec-boundingbox.c' ],
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  include_directories: nnstreamer_inc,
  install: true,
  install_dir: nnstreamer_libdir
)
