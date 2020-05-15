
cc = meson.get_compiler('c')

libovx_dep = cc.find_library('ovxlib') # ovxlib library
libdl_dep = cc.find_library('dl') # dl library


base_deps = [
  glib_dep,
  gobject_dep,
  nnstreamer_dep
]

sources = [
  'tensor_filter_subplugin.c'
]

# TODO: This header files are specified to support Tizen. 
# If you use VIM3/Ubuntu18.04(aarch64) board,
# you need to update the below header folders.
# - '/usr/include/vivante', '/usr/include/VX'
# - https://docs.khadas.com/vim3/HowToUseNpu.html
# - https://dl.khadas.com/repos/vim3/ (Ubuntu 18.04, aarch64, Bionic)
vivante_todo_incs = []
vivante_todo_incs += include_directories(
  '/usr/include/dann',
  '/usr/include/ovx',
  '/usr/include/amlogic-vsi-npu-sdk'
)

vflag = '-ldl'

subplugin_shared = shared_library('nnstreamer_filter_vivante',
  sources,
  include_directories: vivante_todo_incs,
  dependencies: [base_deps, libovx_dep, libdl_dep],
  link_args : vflag,
  install: true,
  install_dir: filter_subplugin_install_dir
)
