# Find flex, configure lex generator
flex_min_version='2.5.31'
flex = find_program('flex', 'win_flex', version: '>=' + flex_min_version, required : false)
if not flex.found()
  if get_option('parser-support').enabled()
    error('flex is required if parser-support is enabled')
  endif
  subdir_done()
endif
flex_cdata = configuration_data()
flex_cdata.set('FLEX', flex.path())
if cc.get_id() == 'msvc'
  flex_cdata.set('FLEX_ARGS', '--nounistd')
else
  flex_cdata.set('FLEX_ARGS', '')
endif

gen_lex = configure_file(input : 'gen_lex.py.in',
  output : 'gen_lex.py',
  configuration : flex_cdata)

# Find bison, configure grammar generator
bison_min_version='2.4'
bison = find_program('bison', 'win_bison', version: '>=' + bison_min_version, required : false)
if not bison.found()
  if get_option('parser-support').enabled()
    error('bison is required if parser-support is enabled')
  endif
  subdir_done()
endif
bison_cdata = configuration_data()
bison_cdata.set('BISON', bison.path())
bison_cdata.set('BISON_ARGS', '')

gen_grammar = configure_file(input : 'gen_grammar.py.in',
  output : 'gen_grammar.py',
  configuration : bison_cdata)

# Custom targets
pymod = import('python')
python3 = pymod.find_installation('python3')

parser = custom_target('parselex',
  input : 'parse.l',
  output : ['lex.priv_gst_parse_yy.c', 'parse_lex.h'],
  command : [python3, gen_lex, '@OUTPUT0@', '@OUTPUT1@', '@INPUT@', 'DUMMY']
)

grammar = custom_target('parsegrammar',
  input : 'grammar.y',
  output : ['grammar.tab.c', 'grammar.tab.h'],
  command : [python3, gen_grammar, '@OUTPUT0@', '@OUTPUT1@', '@INPUT@'],
  depends : [parser],
)

# Top-level parser app
toplevel_srcs = [
  parser,
  grammar,
  'types.c',
  'toplevel.c',
  'convert.c'
]
toplevel_deps = [
  glib_dep,
  gobject_dep
]
toplevel = executable('nnstreamer-parser', toplevel_srcs,
  dependencies: toplevel_deps,
  install: true,
  install_dir: join_paths(get_option('prefix'), get_option('bindir')),
)
