tensor_mux_sources = [
    'gsttensormux.c'
]

tensor_muxOBJ = static_library('tensor_mux',
    tensor_mux_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_muxOBJ

