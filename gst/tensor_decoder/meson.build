tensor_decoder_sources = [
    'tensordec.c',
    'tensordec-plugins.c',
    'tensordec-directvideo.c'    
]

tensor_decoderOBJ = static_library('tensor_decoder',
    tensor_decoder_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_decoderOBJ
