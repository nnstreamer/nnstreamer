tensor_common_sources = [
    'tensor_common.c',
    'tensor_repo.c'
]

tensor_commonOBJ = static_library('tensor_common', 
    tensor_common_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_commonOBJ
