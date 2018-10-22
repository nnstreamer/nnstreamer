pkg_check_modules(opencv_pkg REQUIRED opencv)

ADD_LIBRARY(nnstreamer_customfilter_opencv_scaler SHARED nnstreamer_customfilter_opencv_scaler.cc)
TARGET_LINK_LIBRARIES(nnstreamer_customfilter_opencv_scaler ${pkgs_LIBRARIES} ${opencv_pkg_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_customfilter_opencv_scaler PUBLIC ${pkgs_INCLUDE_DIRS} ${opencv_pkg_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_customfilter_opencv_scaler PUBLIC ${pkgs_CFLAGS_OTHER})

INSTALL(TARGETS nnstreamer_customfilter_opencv_scaler
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)
