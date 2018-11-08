ADD_LIBRARY(nnstreamer_customfilter_average SHARED nnstreamer_customfilter_example_average.c)

TARGET_LINK_LIBRARIES(nnstreamer_customfilter_average ${CUSTOMFILTERDEP})

INSTALL(TARGETS nnstreamer_customfilter_average
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
