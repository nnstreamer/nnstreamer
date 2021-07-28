tensor_transform_sources = [
  'tensor_transform.c'
]

if orcc_support_is_available
  # tensor_transform does not need to generate -dist files.
  orcsrc = 'transform-orc'

  orc_h = custom_target(orcsrc + '.h',
    input: orcsrc + '.orc',
    output: orcsrc + '.h',
    command: orcc_support_orcc_args + ['--header', '-o', '@OUTPUT@', '@INPUT@'])
  orc_c_org = custom_target(orcsrc + '.c.in',
    input: orcsrc + '.orc',
    output: orcsrc + '.c.in',
    command: orcc_support_orcc_args + ['--implementation', '-o', '@OUTPUT@', '@INPUT@'])

## ORC generated codes incur a few warnings. Ignore warnings of generated files.
  script = find_program('ignore_warning.sh')
  orc_c = custom_target(orcsrc + '.c',
    input: orc_c_org,
    output: orcsrc + '.c',
    command: [ script, '@INPUT@', '@OUTPUT@' ])

  nnstreamer_sources += [orc_c, orc_h]
  nnstreamer_internal_deps += declare_dependency(sources: orc_h)
endif

foreach s : tensor_transform_sources
  nnstreamer_sources += join_paths(meson.current_source_dir(), s)
endforeach
