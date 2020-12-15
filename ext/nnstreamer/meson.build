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
    prog_which = find_program('which')
    grpc_cpp_plugin_path = run_command(prog_which, 'grpc_cpp_plugin').stdout().strip()
    if grpc_cpp_plugin_path == ''
      error('cannot find the path of grpc_cpp_plugin.')
    endif

    grpc_pb_gen = generator(pb_comp,
        output: ['@BASENAME@.grpc.pb.h', '@BASENAME@.grpc.pb.cc'],
        arguments : [
          '--proto_path=@CURRENT_SOURCE_DIR@/include',
          '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin_path,
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

if get_option('enable-tizen-sensor')
  tizensensor_registerer_source_files = ['registerer/tizensensor.c']
  tizensensor_registerer_sources = []

  foreach s : tizensensor_registerer_source_files
    tizensensor_registerer_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  tizensensor_lib = shared_library('nnstreamer-tizen-sensor',
    tizensensor_registerer_sources,
    dependencies: tensor_src_tizensensor_dep,
    install: true,
    install_dir: plugins_install_dir
  )

  tizensensor_static = static_library('nnstreamer-tizen-sensor',
    tizensensor_registerer_sources,
    dependencies: tensor_src_tizensensor_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if grpc_support_is_available
  grpc_registerer_source_files = ['registerer/grpc.c']
  grpc_registerer_sources = []

  foreach s : grpc_registerer_source_files
    grpc_registerer_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  grpc_lib = shared_library('nnstreamer-grpc',
    grpc_registerer_sources,
    dependencies: [tensor_src_grpc_dep, tensor_sink_grpc_dep],
    install: true,
    install_dir: plugins_install_dir
  )
endif
