if meson.project_name() != 'nnstreamer'

  project('capi-nnstreamer', 'c',
    version: '0.1',
    license: ['LGPL'],
    meson_version: '>=0.42.0',
    default_options: [
      'werror=true',
      'warning_level=1',
      'c_std=c89'
    ]
  )

  add_project_arguments('-DVERSION="'+meson.project_version()+'"', language: ['c', 'cpp'])

  cc = meson.get_compiler('c')
  cxx = meson.get_compiler('cpp')
endif


inc = include_directories('include')
nninc = include_directories('../../gst')

# Create dependency for tensor_filter_single for single-shot capi
tensor_filter_single_lib = shared_library ('tensor_filter_single_lib',
  join_paths(meson.current_source_dir(), 'src', 'tensor_filter_single.c'),
  dependencies: [glib_dep, gobject_dep],
  include_directories: [inc, nninc],
  install: false,
  version: meson.project_version(),
)

tensor_filter_single_dep = declare_dependency(link_with: tensor_filter_single_lib,
  include_directories: [inc],
)

capi_main = []
capi_main += join_paths(meson.current_source_dir(), 'src', 'nnstreamer-capi-pipeline.c')
capi_main += join_paths(meson.current_source_dir(), 'src', 'nnstreamer-capi-util.c')
capi_main += join_paths(meson.current_source_dir(), 'src', 'nnstreamer-capi-single.c')

capi_devel_main = []
capi_devel_main += join_paths(meson.current_source_dir(), 'include', 'nnstreamer.h')
capi_devel_main += join_paths(meson.current_source_dir(), 'include', 'nnstreamer-single.h')

tizen_deps = []

if (get_option('enable-tizen'))
  message('CAPI is in Tizen mode')
  tizen_deps = [
    dependency('capi-base-common'),
    dependency('capi-system-info'),
    dependency('dlog')
  ]
else
  r = run_command('./modify_nnstreamer_h_for_nontizen.sh')
  if (r.returncode() != 0)
    message('Non-Tizen mode support failed')
  endif
  message('CAPI is in non-Tizen mode')
endif

capi_deps = [
  nnstreamer_dep, glib_dep, gst_dep, gst_app_dep, tizen_deps, tensor_filter_single_dep
]

nnstreamer_capi_lib = shared_library ('capi-nnstreamer',
  capi_main,
  dependencies: capi_deps,
  include_directories: [
    inc,
    nninc
  ],
  install: true,
  version: meson.project_version(),
  install_dir: nnstreamer_libdir,
)
static_library ('capi-nnstreamer',
  capi_main,
  dependencies: capi_deps,
  include_directories: [
    inc,
    nninc
  ],
  install: true,
  install_dir: nnstreamer_libdir,
)

nnstreamer_capi_dep = declare_dependency(link_with: nnstreamer_capi_lib,
  dependencies: capi_deps,
  include_directories: [
    inc,
    nninc
  ]
)

configure_file(input: 'capi-nnstreamer.pc.in', output: 'capi-nnstreamer.pc',
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_conf
)

install_headers(capi_devel_main,
  subdir: 'nnstreamer'
)
