if get_option('enable-tizen-sensor')
  tizensensor_registerer_source_files = files('tizensensor.c')

  tizensensor_lib = shared_library('nnstreamer-tizen-sensor',
    tizensensor_registerer_source_files,
    dependencies: tensor_src_tizensensor_dep,
    install: true,
    install_dir: plugins_install_dir
  )

  tizensensor_static = static_library('nnstreamer-tizen-sensor',
    tizensensor_registerer_source_files,
    dependencies: tensor_src_tizensensor_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
endif

if grpc_support_is_available
  grpc_registerer_source_files = files('grpc.c')

  grpc_lib = shared_library('nnstreamer-grpc',
    grpc_registerer_source_files,
    dependencies: [tensor_src_grpc_dep, tensor_sink_grpc_dep],
    install: true,
    install_dir: plugins_install_dir
  )
endif
