ADD_LIBRARY(nnstreamer_customfilter_average SHARED nnstreamer_customfilter_example_average.c)

INSTALL(TARGETS nnstreamer_customfilter_average
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
