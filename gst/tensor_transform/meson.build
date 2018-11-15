tensor_transform_sources = [
    'tensor_transform.c'
]

tensor_transformOBJ = static_library('tensor_transform',
    tensor_transform_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_transformOBJ
