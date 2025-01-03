if protobuf_support_is_available
  # Don't generate proto files twice
  nns_protobuf_lib = shared_library('nnstreamer_protobuf',
    sources: [pb_gen_src, 'nnstreamer_protobuf.cc'],
    dependencies: [protobuf_dep, nnstreamer_dep],
    install: true,
    install_dir: nnstreamer_libdir
  )
  nns_protobuf_dep = declare_dependency(
    link_with: nns_protobuf_lib,
    dependencies: protobuf_dep,
    include_directories: nns_protobuf_lib.private_dir_include()
  )

  protobuf_util_dep = declare_dependency(
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, nns_protobuf_dep],
    include_directories: include_directories('.')
  )
endif

if grpc_support_is_available
  # Do NOT add grpc_support_deps to avoid error while compiling .c files (-Wnon-virtual-dtor)
  grpc_util_deps = [nnstreamer_dep, glib_dep, gst_dep]
  if cxx.has_argument('-Wno-unused-parameter')
    grpc_util_deps += declare_dependency(compile_args: ['-Wno-unused-parameter'])
  endif

  grpc_util_lib = shared_library('nnstreamer-grpc-util',
    sources: ['nnstreamer_grpc_common.cc'],
    dependencies: [grpc_util_deps, grpc_support_deps],
    install: true,
    install_dir: nnstreamer_libdir
  )

  grpc_util_dep = declare_dependency(
    link_with: grpc_util_lib,
    dependencies: grpc_util_deps,
    include_directories: include_directories('.')
  )

  if protobuf_support_is_available
    nns_protobuf_grpc_src = ['nnstreamer_grpc_protobuf.cc']

    # Don't generate proto files twice
    nns_protobuf_grpc_lib = shared_library ('nnstreamer_grpc_protobuf',
      sources : [grpc_pb_gen_src, nns_protobuf_grpc_src],
      dependencies : [grpc_util_dep, grpc_support_deps, nns_protobuf_dep],
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
      dependencies : [grpc_util_dep, grpc_support_deps, flatbuf_dep],
      install: true,
      install_dir: nnstreamer_libdir
    )
  else
    warning('gRPC/Flatbuf is not supported')
  endif
endif

if have_python3
  nnstreamer_python3_deps = [python3_dep, libdl_dep, glib_dep, nnstreamer_single_dep]
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
