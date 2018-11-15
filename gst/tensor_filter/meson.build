tensor_filter_sources = [
    'tensor_filter.c',
    'tensor_filter_custom.c'
]

tensor_filterOBJ = static_library('tensor_filter',
    tensor_filter_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_filterOBJ
