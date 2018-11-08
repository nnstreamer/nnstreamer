ADD_LIBRARY(nnstreamer_customfilter_scaler SHARED nnstreamer_customfilter_example_scaler.c)

TARGET_LINK_LIBRARIES(nnstreamer_customfilter_scaler ${CUSTOMFILTERDEP})

ADD_LIBRARY(nnstreamer_customfilter_scaler_allocator SHARED nnstreamer_customfilter_example_scaler_allocator.c)

TARGET_LINK_LIBRARIES(nnstreamer_customfilter_scaler_allocator ${CUSTOMFILTERDEP})

INSTALL(TARGETS nnstreamer_customfilter_scaler nnstreamer_customfilter_scaler_allocator
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${LIB_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
