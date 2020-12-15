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
  grpc_common_src = ['nnstreamer_grpc_common.cc']
  grpc_util_sources = []

  foreach s : grpc_common_src
    grpc_util_sources += join_paths(meson.current_source_dir(), s)
  endforeach

  grpc_util_dep = declare_dependency(
    sources: grpc_util_sources,
    dependencies: [nnstreamer_dep, glib_dep, gst_dep, libdl_dep, grpc_support_deps],
    include_directories: include_directories('.')
  )

  if protobuf_support_is_available
    nns_protobuf_grpc_src = ['nnstreamer_grpc_protobuf.cc']
    nns_protobuf_grpc_sources = []
    foreach s : nns_protobuf_grpc_src
      nns_protobuf_grpc_sources += join_paths(meson.current_source_dir(), s)
    endforeach

    # Don't generate proto files twice
    nns_protobuf_grpc_lib = shared_library ('nnstreamer_grpc_protobuf',
      sources : [grpc_pb_gen_src, nns_protobuf_grpc_sources],
      dependencies : [grpc_util_dep, nns_protobuf_dep],
      install: true,
      install_dir: nnstreamer_libdir
    )
    nns_protobuf_grpc_dep = declare_dependency(
      link_with: nns_protobuf_grpc_lib,
      dependencies : [grpc_util_dep, nns_protobuf_dep],
      include_directories: nns_protobuf_grpc_lib.private_dir_include()
    )
  else
    warning('gRPC/Protobuf is not supported')
  endif

  if flatbuf_support_is_available
    nns_flatbuf_grpc_src = ['nnstreamer_grpc_flatbuf.cc']
    nns_flatbuf_grpc_sources = []
    foreach s : nns_flatbuf_grpc_src
      nns_flatbuf_grpc_sources += join_paths(meson.current_source_dir(), s)
    endforeach

    nns_flatbuf_grpc_lib = shared_library ('nnstreamer_grpc_flatbuf',
      sources : [grpc_fb_gen_src, nns_flatbuf_grpc_sources],
      dependencies : [grpc_util_dep, flatbuf_dep],
      install: true,
      install_dir: nnstreamer_libdir
    )
    nns_flatbuf_grpc_dep = declare_dependency(
      link_with: nns_flatbuf_grpc_lib,
      dependencies : [grpc_util_dep, flatbuf_dep],
      include_directories: nns_flatbuf_grpc_lib.private_dir_include()
    )
  else
    warning('gRPC/Flatbuf is not supported')
  endif
endif
