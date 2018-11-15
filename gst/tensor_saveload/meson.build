tensor_saveload_sources = [
    'tensor_load.c'
]

tensor_saveloadOBJ = static_library('tensor_saveload',
    tensor_saveload_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_saveloadOBJ
