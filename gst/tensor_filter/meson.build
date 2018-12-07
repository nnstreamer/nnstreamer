tensor_filter_sources = [
    'tensor_filter.c',
    'tensor_filter_custom.c'
]

tensor_filter_args = [nnstreamer_base_args]

if get_option('ENABLE_TENSORFLOW_LITE')
    tensor_filter_sources += 'tensor_filter_tensorflow_lite.c'
    dependencies += dependency('tensorflow-lite')
    tensor_filter_args += ['-DENABLE_TENSORFLOW_LITE']

    tensor_filter_tflitecoreOBJ = static_library('tensor_filter_tflitecore',
        'tensor_filter_tensorflow_lite_core.cc',
        dependencies: [
            dependencies
        ],
        include_directories: inc,
        cpp_args: tensor_filter_args + ['-Wno-sign-compare']
    ).extract_all_objects()

    objects += tensor_filter_tflitecoreOBJ
endif

if get_option('ENABLE_TENSORFLOW')
    tensor_filter_sources += 'tensor_filter_tensorflow.c'
    dependencies += dependency('tensorflow')
    tensor_filter_args += ['-DENABLE_TENSORFLOW']

    tensor_filter_tfcoreOBJ = static_library('tensor_filter_tfcore',
        'tensor_filter_tensorflow_core.cc',
        dependencies: [
            dependencies
        ],
        include_directories: inc,
        cpp_args: tensor_filter_args + ['-Wno-sign-compare']
    ).extract_all_objects()

    objects += tensor_filter_tflitecoreOBJ
endif

tensor_filterOBJ = static_library('tensor_filter',
    tensor_filter_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    c_args: tensor_filter_args
).extract_all_objects()

objects += tensor_filterOBJ
