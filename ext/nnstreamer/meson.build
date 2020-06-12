if flatbuf_support_is_available
  # Compile flatbuffers schema file
  fb_gen = generator(fb_comp, output : '@BASENAME@_generated.h',
                 arguments : ['--cpp', '-o', '@BUILD_DIR@', '@INPUT@'])
  fb_gen_src = fb_gen.process('./include/nnstreamer.fbs')
endif

if protobuf_support_is_available
  pb_gen = generator(pb_comp,
      output: ['@BASENAME@.pb.h', '@BASENAME@.pb.cc'],
      arguments : [
        '--proto_path=@CURRENT_SOURCE_DIR@/include',
        '--cpp_out=@BUILD_DIR@',
        '@INPUT@'
      ]
  )
  pb_gen_src = pb_gen.process('./include/nnstreamer.proto')
endif

subdir('extra')
subdir('tensor_decoder')
subdir('tensor_filter')
subdir('tensor_source')
subdir('tensor_converter')
