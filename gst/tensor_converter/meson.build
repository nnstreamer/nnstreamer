tensor_converter_sources = [
    'tensor_converter.c'
]

tensor_converterOBJ = static_library('tensor_converter',
    tensor_converter_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_converterOBJ
