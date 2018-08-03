ADD_LIBRARY(nnstreamer_customfilter_average SHARED nnstreamer_customfilter_example_average.c)

TARGET_LINK_LIBRARIES(nnstreamer_customfilter_average ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_customfilter_average PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_customfilter_average PUBLIC ${pkgs_CFLAGS_OTHER})

INSTALL(TARGETS nnstreamer_customfilter_average
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
