dependencies += dependency('opencv')

nnstreamer_customfilter_opencv_scaler = shared_library('nnstreamer_customfilter_opencv_scaler',
    'nnstreamer_customfilter_opencv_scaler.cc',
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)

nnstreamer_customfilter_opencv_average = shared_library('nnstreamer_customfilter_opencv_average',
    'nnstreamer_customfilter_opencv_average.cc',
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)
