# direct video
decoder_sub_direct_video_sources = [
  'tensordec-directvideo.c'
]

nnstreamer_decoder_direct_video_sources = []
foreach s : decoder_sub_direct_video_sources
  nnstreamer_decoder_direct_video_sources += join_paths(meson.current_source_dir(), s)
endforeach

shared_library('nnstreamer_decoder_direct_video',
  nnstreamer_decoder_direct_video_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_direct_video',
  nnstreamer_decoder_direct_video_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# image labeling
decoder_sub_image_labeling_sources = [
  'tensordec-imagelabel.c'
]

nnstreamer_decoder_image_labeling_sources = []
foreach s : decoder_sub_image_labeling_sources
  nnstreamer_decoder_image_labeling_sources += join_paths(meson.current_source_dir(), s)
endforeach

shared_library('nnstreamer_decoder_image_labeling',
  nnstreamer_decoder_image_labeling_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_image_labeling',
  nnstreamer_decoder_image_labeling_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)

# bounding boxes
decoder_sub_bounding_boxes_sources = [
  'tensordec-boundingbox.c'
]

nnstreamer_decoder_bounding_boxes_sources = []
foreach s : decoder_sub_bounding_boxes_sources
  nnstreamer_decoder_bounding_boxes_sources += join_paths(meson.current_source_dir(), s)
endforeach

shared_library('nnstreamer_decoder_bounding_boxes',
  nnstreamer_decoder_bounding_boxes_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: decoder_subplugin_install_dir
)
static_library('nnstreamer_decoder_bounding_boxes',
  nnstreamer_decoder_bounding_boxes_sources,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep],
  install: true,
  install_dir: nnstreamer_libdir
)
