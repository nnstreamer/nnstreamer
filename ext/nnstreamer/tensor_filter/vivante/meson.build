libovx_dep = cc.find_library('ovxlib') # ovxlib library

sources = [
  'tensor_filter_vivante.c'
]

# TODO: This header files are specified to support Tizen.
# If you use VIM3/Ubuntu18.04(aarch64) board,
# you need to update the below header folders.
# - '/usr/include/vivante', '/usr/include/VX'
# - https://docs.khadas.com/vim3/HowToUseNpu.html
# - https://dl.khadas.com/repos/vim3/ (Ubuntu 18.04, aarch64, Bionic)
vivante_incs = include_directories(
  '/usr/include/ovx',
  '/usr/include/amlogic-vsi-npu-sdk'
)

shared_library('nnstreamer_filter_vivante',
  sources,
  include_directories: vivante_incs,
  dependencies: [nnstreamer_single_dep, libovx_dep, libdl_dep],
  link_args : '-ldl',
  install: true,
  install_dir: filter_subplugin_install_dir
)
