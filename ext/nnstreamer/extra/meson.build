# Don't generate protobuf/flatbuf files twice
if protobuf_support_is_available
  protobuf_util_deps = [nnstreamer_dep, glib_dep, gst_dep, protobuf_dep]
  protobuf_util_lib = shared_library('nnstreamer_protobuf',
    sources: [pb_gen_src, 'nnstreamer_protobuf.cc'],
    dependencies: protobuf_util_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  protobuf_util_dep = declare_dependency(
    link_with: protobuf_util_lib,
    dependencies: protobuf_util_deps,
    include_directories: [include_directories('.'), protobuf_util_lib.private_dir_include()]
  )
endif

if flatbuf_support_is_available
  flatbuf_util_deps = [nnstreamer_dep, glib_dep, gst_dep, flatbuf_dep]
  flatbuf_util_lib = shared_library('nnstreamer_flatbuf',
    sources: [fb_gen_src],
    dependencies: flatbuf_util_deps,
    install: true,
    install_dir: nnstreamer_libdir
  )

  flatbuf_util_dep = declare_dependency(
    link_with: flatbuf_util_lib,
    dependencies: flatbuf_util_deps,
    include_directories: [include_directories('.'), flatbuf_util_lib.private_dir_include()]
  )
endif

if grpc_support_is_available
  # Do NOT add grpc_support_deps to avoid error while compiling .c files (-Wnon-virtual-dtor)
  grpc_util_deps = [nnstreamer_dep, glib_dep, gst_dep]
  if cxx.has_argument('-Wno-unused-parameter')
    grpc_util_deps += declare_dependency(compile_args: ['-Wno-unused-parameter'])
  endif

  grpc_util_lib = shared_library('nnstreamer_grpc',
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

  # Don't generate protobuf/flatbuf files twice
  if protobuf_support_is_available
    nns_protobuf_grpc_lib = shared_library ('nnstreamer_grpc_protobuf',
      sources : [grpc_pb_gen_src, 'nnstreamer_grpc_protobuf.cc'],
      dependencies : [grpc_util_dep, grpc_support_deps, protobuf_util_dep],
      install: true,
      install_dir: nnstreamer_libdir
    )
  else
    warning('gRPC/Protobuf is not supported')
  endif

  if flatbuf_support_is_available
    nns_flatbuf_grpc_lib = shared_library ('nnstreamer_grpc_flatbuf',
      sources : [grpc_fb_gen_src, 'nnstreamer_grpc_flatbuf.cc'],
      dependencies : [grpc_util_dep, grpc_support_deps, flatbuf_util_dep],
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
