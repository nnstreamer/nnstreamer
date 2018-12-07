# If you are using Ubuntu/Xenial, Do "force-version" on meson to get 0.40
# If you are using Tizen 5.0+ or Ubuntu/Bionix+, you don't need to mind meson version.

project('nnstreamer', 'c', 'cpp',
    version: '0.1.0',
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

glib_dep = dependency('glib-2.0')
gst_dep = dependency('gstreamer-' + gst_api_verision)
gst_base_dep = dependency('gstreamer-base-' + gst_api_verision)
gst_controller_dep = dependency('gstreamer-controller-' + gst_api_verision)
gst_video_dep = dependency('gstreamer-video-' + gst_api_verision)
gst_audio_dep = dependency('gstreamer-audio-' + gst_api_verision)
gst_app_dep = dependency('gstreamer-app-' + gst_api_verision)
gst_check_dep = dependency('gstreamer-check-' + gst_api_verision)

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

dependencies = [
    glib_dep,
    gst_dep,
    gst_base_dep,
    gst_controller_dep,
    gst_video_dep,
    gst_audio_dep,
    # required 'false' mean is used only when needed.
    cc.find_library('m', required: false), # cmath library
    cc.find_library('dl', required: false), # DL library
    cxx.find_library('pthread', required: false) # pthread for tensor_filter_tensorflow_lite
]

nnstreamer_base_args = []

inc = include_directories('gst/nnstreamer')

# Orc
have_orcc = false
orcc_args = []

orc_dep = dependency('orc-0.4', version : '>= 0.4.24', required : true)
orcc = find_program('orcc', required : true)
if orc_dep.found() and orcc.found()
  have_orcc = true
  orcc_args = [orcc, '--include', 'glib.h']
  dependencies += [orc_dep]
  nnstreamer_base_args += ['-DHAVE_ORC=1']
else
  message('Cannot find orc library')
  nnstreamer_base_args += ['-DDISABLE_ORC=1']
endif

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
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)
nnstreamer = static_library('nnstreamer',
    nnstreamer_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc,
    objects: objects
)
