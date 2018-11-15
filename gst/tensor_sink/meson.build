tensor_sink_sources = [
    'tensor_sink.c'
]

tensor_sinkOBJ = static_library('tensor_sink',
    tensor_sink_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_sinkOBJ
