testtensors_sources = [
  'gsttesttensors.c'
]

testtensors_lib = library('testtensors',
  testtensors_sources,
  dependencies: [gst_dep, nnstreamer_dep]
)

tensorscheck_sources = [
    'gsttensorscheck.c'
]

tensorscheck_lib = library('tensorscheck',
  tensorscheck_sources,
  dependencies: [gst_dep, nnstreamer_dep]
)
