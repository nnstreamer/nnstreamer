tensor_src_sources = [
]

gst18_dep = dependency('gstreamer-' + gst_api_verision, version : '>=1.8', required : false)
if gst18_dep.found()
  tensor_src_sources += 'tensor_src_iio.c'
else
  message('tensor_src_iio requires GStreamer >= 1.8. Skipping it')
endif

foreach s : tensor_src_sources
  if build_platform != 'macos'
    nnstreamer_sources += join_paths(meson.current_source_dir(), s)
  endif
endforeach
