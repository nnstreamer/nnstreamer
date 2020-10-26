if protobuf_support_is_available
  # Don't generate .proto file twice
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

  nns_protobuf_util_sources = []
  foreach s : protobuf_util_sources
    nns_protobuf_util_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  protobuf_util_dep = declare_dependency(
    sources: nns_protobuf_util_sources,
    dependencies: protobuf_util_deps,
    include_directories: include_directories('.')
  )
endif

if grpc_support_is_available
  if not protobuf_support_is_available
    error('failed to resolve dependency: gRPC requires libprotobuf.')
  endif

  # Don't generate .proto file twice
  nns_protobuf_grpc_lib = shared_library ('nnstreamer_protobuf_grpc',
    sources : grpc_pb_gen_src,
    dependencies : [grpc_support_deps, nns_protobuf_dep],
    install: true,
    install_dir: nnstreamer_libdir
  )
  nns_protobuf_grpc_dep = declare_dependency(
    link_with: nns_protobuf_grpc_lib,
    dependencies : [grpc_support_deps, nns_protobuf_dep],
    include_directories: nns_protobuf_grpc_lib.private_dir_include()
  )

  grpc_protobuf_util_sources = ['nnstreamer_protobuf_grpc.cc']
  grpc_protobuf_util_deps = [nnstreamer_dep, glib_dep, gst_dep, nns_protobuf_grpc_dep]

  nns_grpc_protobuf_util_sources = []
  foreach s : grpc_protobuf_util_sources
    nns_grpc_protobuf_util_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  grpc_protobuf_util_dep = declare_dependency(
    sources: nns_grpc_protobuf_util_sources,
    dependencies: grpc_protobuf_util_deps,
    include_directories: include_directories('.')
  )
endif
