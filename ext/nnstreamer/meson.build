if flatbuf_support_is_available
  # Compile flatbuffers schema file
  fb_gen = generator(fb_comp, output : '@BASENAME@_generated.h',
                 arguments : ['--cpp', '-o', '@CURRENT_SOURCE_DIR@/include', '@INPUT@'])
  fb_gen_src = fb_gen.process('./include/nnstreamer.fbs')
endif

subdir('tensor_decoder')
subdir('tensor_filter')
subdir('tensor_source')
subdir('tensor_converter')
