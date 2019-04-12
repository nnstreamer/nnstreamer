if meson.project_name() != 'nnstreamer'

  project('capi-nnstreamer', 'c',
    version: '0.1',
    license: ['Apache2'],
    meson_version: '>=0.40.0',
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


capi_main = []
capi_main += join_paths(meson.current_source_dir(), 'src', 'tizen-api-pipeline.c')

capi_devel_main = []
capi_devel_main += join_paths(meson.current_source_dir(), 'include', 'tizen-api.h')

inc = include_directories('include')
nninc = include_directories('../gst')

capi_base_common_dep = dependency('capi-base-common')
dlog_dep = dependency('dlog')
tizen_deps = [
  capi_base_common_dep,
  dlog_dep
]

tizen_capi_lib = shared_library ('capi-nnstreamer',
  capi_main,
  dependencies: [nnstreamer_dep, glib_dep, gst_dep, gst_app_dep, tizen_deps],
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
  dependencies: [nnstreamer_dep, glib_dep, gst_dep, gst_app_dep, tizen_deps],
  include_directories: [
    inc,
    nninc
  ],
  install: true,
  install_dir: nnstreamer_libdir,
)

tizen_capi_dep = declare_dependency(link_with: tizen_capi_lib,
  include_directories: [
    inc,
    nninc
  ]
)

configure_file(input: 'capi-nnstreamer.pc.in', output: 'capi-nnstreamer.pc',
  install: true,
  install_dir: join_paths(nnstreamer_libdir, 'pkgconfig'),
  configuration: nnstreamer_conf
)

install_headers(capi_devel_main,
  subdir: 'nnstreamer'
)
