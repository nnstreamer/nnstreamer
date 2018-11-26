tensor_repopush_sources = [
    'tensor_repopush.c'
]

tensor_repopushOBJ = static_library('tensor_repopush',
    tensor_repopush_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_repopushOBJ
