if protobuf_support_is_available
  # Don't generate proto files twice
  nns_protobuf_lib = shared_library('nnstreamer_protobuf',
    sources: pb_gen_src,
    dependencies: protobuf_dep,
    install: true,
    install_dir: nnstreamer_libdir
  )
  nns_protobuf_dep = declare_dependency(
    link_with: nns_protobuf_lib,
    dependencies: protobuf_dep,
    include_directories: nns_protobuf_lib.private_dir_include()
  )

  protobuf_util_sources = ['nnstreamer_protobuf.cc']
  protobuf_util_deps = [nnstreamer_dep, glib_dep, gst_dep, nns_protobuf_dep]

  protobuf_util_dep = declare_dependency(
    sources: protobuf_util_sources,
    dependencies: protobuf_util_deps,
    include_directories: include_directories('.')
  )
endif

if grpc_support_is_available
  grpc_common_src = ['nnstreamer_grpc_common.cc']
  ignore_unused_param_auto_generated_files = declare_dependency(compile_args: ['-Wno-unused-parameter'])

  grpc_util_dep = declare_dependency(
    sources: grpc_common_src,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, libdl_dep, grpc_support_deps, ignore_unused_param_auto_generated_files],
    include_directories: include_directories('.')
  )

  if protobuf_support_is_available
    nns_protobuf_grpc_src = ['nnstreamer_grpc_protobuf.cc']

    # Don't generate proto files twice
    nns_protobuf_grpc_lib = shared_library ('nnstreamer_grpc_protobuf',
      sources : [grpc_pb_gen_src, nns_protobuf_grpc_src],
      dependencies : [grpc_util_dep, nns_protobuf_dep, ignore_unused_param_auto_generated_files],
      install: true,
      install_dir: nnstreamer_libdir
    )
  else
    warning('gRPC/Protobuf is not supported')
  endif

  if flatbuf_support_is_available
    nns_flatbuf_grpc_src = ['nnstreamer_grpc_flatbuf.cc']

    nns_flatbuf_grpc_lib = shared_library ('nnstreamer_grpc_flatbuf',
      sources : [grpc_fb_gen_src, nns_flatbuf_grpc_src],
      dependencies : [grpc_util_dep, flatbuf_dep, ignore_unused_param_auto_generated_files],
      install: true,
      install_dir: nnstreamer_libdir
    )
  else
    warning('gRPC/Flatbuf is not supported')
  endif
endif

if have_python3
  nnstreamer_python3_deps = [python3_dep, libdl_dep, glib_dep, gst_dep, nnstreamer_dep]
  nnstreamer_python3_src = ['nnstreamer_python3_helper.cc']

  nnstreamer_python3_helper = shared_library('nnstreamer_python3',
    sources: nnstreamer_python3_src,
    name_prefix: '',
    dependencies: nnstreamer_python3_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  nnstreamer_python3_helper_dep = declare_dependency(
    link_with: nnstreamer_python3_helper,
    dependencies: nnstreamer_python3_deps,
    include_directories: include_directories('.')
  )
endif
