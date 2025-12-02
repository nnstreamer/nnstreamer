if flatbuf_support_is_available
  # Compile flatbuffers schema file
  fb_gen = generator(flatc, output : '@BASENAME@_generated.h',
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

if grpc_support_is_available
  if protobuf_support_is_available
    # gRPC/Protobuf
    prog_grpc_cpp_plugin = find_program('grpc_cpp_plugin', required : true)
    grpc_pb_gen = generator(pb_comp,
        output: ['@BASENAME@.grpc.pb.h', '@BASENAME@.grpc.pb.cc'],
        arguments : [
          '--proto_path=@CURRENT_SOURCE_DIR@/include',
          '--plugin=protoc-gen-grpc=' + prog_grpc_cpp_plugin.full_path(),
          '--grpc_out=@BUILD_DIR@',
          '@INPUT@'
        ]
      )
    grpc_pb_gen_src = grpc_pb_gen.process('./include/nnstreamer.proto')
  endif

  if flatbuf_support_is_available
    # gRPC/Flatbuf
    grpc_fb_gen = generator(flatc,
        output : ['@BASENAME@_generated.h', '@BASENAME@.grpc.fb.h', '@BASENAME@.grpc.fb.cc'],
        arguments : ['--cpp', '--grpc', '-o', '@BUILD_DIR@', '@INPUT@']
      )
    grpc_fb_gen_src = grpc_fb_gen.process('./include/nnstreamer.fbs')
  endif
endif

subdir('extra')
subdir('tensor_decoder')
subdir('tensor_filter')
subdir('tensor_source')
subdir('tensor_converter')
subdir('tensor_sink')
subdir('registerer')
