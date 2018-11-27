tensor_reposink_sources = [
    'tensor_reposink.c'
]

tensor_reposinkOBJ = static_library('tensor_reposink',
    tensor_reposink_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_reposinkOBJ
