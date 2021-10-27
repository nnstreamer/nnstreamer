# NNStreamer configuration checker, "nnstreamer-check"
if get_option('enable-nnstreamer-check')
  subdir('confchk')
endif

# Gst/NNS string pipeline desciption <--> pbtxt pipeline description
# for pbtxt pipeline WYSIWYG tools.
if get_option('enable-pbtxt-converter')
  if (build_platform == 'macos')
    message ('pbtxt converter / parser is not yet compatible with macos. Disabling it')
  else
    subdir('parser')
  endif
endif
