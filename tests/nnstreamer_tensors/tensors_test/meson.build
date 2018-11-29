testtensors_sources = [
    'gsttesttensors.c'
]

testtensors = shared_library('testtensors', 
    testtensors_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: tensor_commonOBJ
)

tensorscheck_sources = [
    'gsttensorscheck.c'
]

tensorscheckOBJ = shared_library('tensorscheck',
    tensorscheck_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: tensor_commonOBJ
)
