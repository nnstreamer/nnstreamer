tensor_transform_sources = [
    'tensor_transform.c'
]

orcsrc = 'transform-orc'
if have_orcc
  # tensor_transform does not need to generate -dist files.
  orc_h = custom_target(orcsrc + '.h',
    input : orcsrc + '.orc',
    output : orcsrc + '.h',
    command : orcc_args + ['--header', '-o', '@OUTPUT@', '@INPUT@'])
  orc_c = custom_target(orcsrc + '.c',
    input : orcsrc + '.orc',
    output : orcsrc + '.c',
    command : orcc_args + ['--implementation', '-o', '@OUTPUT@', '@INPUT@'])
  tensor_transform_sources += [orc_c, orc_h]
endif

tensor_transformOBJ = static_library('tensor_transform',
    tensor_transform_sources,
    c_args: nnstreamer_base_args,
    dependencies: [
        dependencies
    ],
    include_directories: inc
).extract_all_objects()

objects += tensor_transformOBJ
