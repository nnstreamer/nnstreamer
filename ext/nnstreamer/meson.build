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

subdir('extra')
subdir('tensor_decoder')
subdir('tensor_filter')
subdir('tensor_source')
subdir('tensor_converter')

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
