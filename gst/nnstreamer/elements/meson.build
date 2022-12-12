nnstreamer_sources += files(
  'gsttensor_aggregator.c',
  'gsttensor_converter.c',
  'gsttensor_crop.c',
  'gsttensor_debug.c',
  'gsttensor_decoder.c',
  'gsttensor_demux.c',
  'gsttensor_if.c',
  'gsttensor_merge.c',
  'gsttensor_mux.c',
  'gsttensor_rate.c',
  'gsttensor_repo.c',
  'gsttensor_reposink.c',
  'gsttensor_reposrc.c',
  'gsttensor_sink.c',
  'gsttensor_sparsedec.c',
  'gsttensor_sparseenc.c',
  'gsttensor_sparseutil.c',
  'gsttensor_split.c',
  'gsttensor_transform.c'
)

# gsttensorsrc
if tensor_src_iio_build
  nnstreamer_sources += files('gsttensor_srciio.c')
endif

# gsttensortransform
if orcc_support_is_available
  # tensor_transform does not need to generate -dist files.
  orcsrc = 'nnstreamer-orc'

  orc_h = custom_target(orcsrc + '.h',
    input: orcsrc + '.orc',
    output: orcsrc + '.h',
    command: orcc_support_orcc_args + ['--header', '-o', '@OUTPUT@', '@INPUT@'])
  orc_c_org = custom_target(orcsrc + '.c.in',
    input: orcsrc + '.orc',
    output: orcsrc + '.c.in',
    command: orcc_support_orcc_args + ['--implementation', '-o', '@OUTPUT@', '@INPUT@'])

  # ORC generated codes incur a few warnings. Ignore warnings of generated files.
  script = find_program('ignore_warning.sh')
  orc_c = custom_target(orcsrc + '.c',
    input: orc_c_org,
    output: orcsrc + '.c',
    command: [ script, '@INPUT@', '@OUTPUT@' ])

  nnstreamer_sources += [orc_c, orc_h]
  nnstreamer_internal_deps += declare_dependency(sources: orc_h)
endif
