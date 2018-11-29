# If you are using Ubuntu/Xenial, Do "force-version" on meson to get 0.40
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
    version: '0.0.3',
    license: ['LGPL'],
    meson_version: '>=0.40.0',
    default_options: [
        'warning_level=1',
        'c_std=c89',
        'cpp_std=c++11'
    ]
)

add_project_arguments('-DVERSION="'+meson.project_version()+'"', language: 'c')

gst_api_verision = '1.0'

glib = dependency('glib-2.0')
gstreamer = dependency('gstreamer-' + gst_api_verision)
gstreamer_base = dependency('gstreamer-base-' + gst_api_verision)
gstreamer_controller = dependency('gstreamer-controller-' + gst_api_verision)
gstreamer_video = dependency('gstreamer-video-' + gst_api_verision)
gstreamer_audio = dependency('gstreamer-audio-' + gst_api_verision)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

dependencies = [
    glib,
    gstreamer,
    gstreamer_base,
    gstreamer_controller,
    gstreamer_video,
    gstreamer_audio,
    # required 'false' mean is used only when needed.
    cc.find_library('m', required: false), # cmath library
    cc.find_library('dl', required: false), # DL library
    cxx.find_library('pthread', required: false) # pthread for tensor_filter_tensorflow_lite
]

inc = include_directories('gst/nnstreamer')

projects = [
	'nnstreamer',
	'tensor_aggregator',
	'tensor_converter',
	'tensor_decoder',
	'tensor_demux',
	'tensor_merge',
	'tensor_mux',
	'tensor_saveload',
	'tensor_sink',
	'tensor_split',
	'tensor_transform',
	'tensor_filter',
	'tensor_reposrc',
	'tensor_reposink'
]

objects = []
foreach p : projects
    subdir('gst/'+ p)
endforeach

subdir('nnstreamer_example')

if get_option('ENABLE_TEST')
    subdir('tests')
endif

nnstreamer_sources = [
    'gst/nnstreamer/nnstreamer.c'
]

# "both_libraries" are supported from 0.46.
nnstreamer = shared_library('nnstreamer',
    nnstreamer_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)
nnstreamer = static_library('nnstreamer',
    nnstreamer_sources,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)
