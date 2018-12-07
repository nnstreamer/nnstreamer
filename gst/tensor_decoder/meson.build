tensor_decoder_sources = [
	'tensordec.c',
	'tensordec-plugins.c',
	'tensordec-directvideo.c',
	'tensordec-imagelabel.c',
	'tensordec-boundingbox.c'
]

tensor_decoderOBJ = static_library('tensor_decoder',
    tensor_decoder_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_decoderOBJ
