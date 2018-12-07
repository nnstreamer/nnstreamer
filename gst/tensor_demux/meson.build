tensor_demux_sources = [
    'gsttensordemux.c'
]

tensor_demuxOBJ = static_library('tensor_demux',
    tensor_demux_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_demuxOBJ
