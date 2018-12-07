testtensors_sources = [
  'gsttesttensors.c'
]

testtensorsOBJ = shared_library('testtensors',
  testtensors_sources,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: tensor_commonOBJ
)

tensorscheck_sources = [
    'gsttensorscheck.c'
]

tensorscheckOBJ = shared_library('tensorscheck',
  tensorscheck_sources,
  dependencies: [nnstreamer_deps],
  include_directories: nnstreamer_inc,
  objects: tensor_commonOBJ
)
