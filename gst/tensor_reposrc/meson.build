tensor_reposrc_sources = [
    'tensor_reposrc.c'
]

tensor_reposrcOBJ = static_library('tensor_reposrc',
    tensor_reposrc_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_reposrcOBJ
