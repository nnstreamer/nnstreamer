subdir('custom_example_passthrough')
subdir('custom_example_scaler')
subdir('custom_example_average')
if get_option('enable-opencv-test')
   subdir('custom_example_opencv')
   sed=find_program('sed')
   r = run_command(sed,'-i', 's|^TEST_OPENCV=.*|TEST_OPENCV=\"YES\"|g', meson.source_root()+'/tests/nnstreamer_filter_custom/runTest.sh')
   if r.returncode() != 0
      message('ERROR: Cannot Convert ${TEST_OEPNCV} to "YES"')
   endif
else
   sed=find_program('sed')
   r = run_command(sed,'-i', 's|^TEST_OPENCV=.*|TEST_OPENCV=\"NO\"|g', meson.source_root()+'/tests/nnstreamer_filter_custom/runTest.sh')
   if r.returncode() != 0
      message('ERROR: Cannot Convert ${TEST_OEPNCV} to "NO"')
   endif
endif
subdir('custom_example_RNN')
subdir('custom_example_LSTM')
