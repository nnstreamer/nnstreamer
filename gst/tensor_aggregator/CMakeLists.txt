ADD_LIBRARY(tensor_aggregator SHARED tensor_aggregator.c)
ADD_LIBRARY(tensor_aggregatorStatic STATIC tensor_aggregator.c)

TARGET_LINK_LIBRARIES(tensor_aggregator ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(tensor_aggregator PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(tensor_aggregator PUBLIC ${pkgs_CFLAGS_OTHER})
TARGET_LINK_LIBRARIES(tensor_aggregatorStatic ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(tensor_aggregatorStatic PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(tensor_aggregatorStatic PUBLIC ${pkgs_CFLAGS_OTHER})

INSTALL(TARGETS tensor_aggregator tensor_aggregatorStatic
	RUNTIME DESTINATION ${EXEC_PREFIX}
	LIBRARY DESTINATION ${GST_INSTALL_DIR}
	ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
	)
