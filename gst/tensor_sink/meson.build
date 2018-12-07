tensor_sink_sources = [
    'tensor_sink.c'
]

tensor_sinkOBJ = static_library('tensor_sink',
    tensor_sink_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_sinkOBJ
