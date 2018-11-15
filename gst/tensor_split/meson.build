tensor_split_sources = [
    'gsttensorsplit.c'
]

tensor_splitOBJ = static_library('tensor_split',
    tensor_split_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_splitOBJ
