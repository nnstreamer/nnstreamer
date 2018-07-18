CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# demo app
ADD_EXECUTABLE(nnstreamer_example_filter nnstreamer_example_filter.c)

TARGET_LINK_LIBRARIES(nnstreamer_example_filter ${apppkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_filter PUBLIC ${apppkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_filter PUBLIC ${apppkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_example_filter RUNTIME DESTINATION ${EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
