tensor_merge_sources = [
    'gsttensormerge.c'
]

tensor_mergeOBJ = static_library('tensor_merge',
    tensor_merge_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc    
).extract_all_objects()

objects += tensor_mergeOBJ
