# example 1
ADD_EXECUTABLE(nnstreamer_sink_example nnstreamer_sink_example.c)

TARGET_LINK_LIBRARIES(nnstreamer_sink_example ${apppkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_sink_example PUBLIC ${apppkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_sink_example PUBLIC ${apppkgs_CFLAGS_OTHER})

# example 2
ADD_EXECUTABLE(nnstreamer_sink_example_play nnstreamer_sink_example_play.c)

TARGET_LINK_LIBRARIES(nnstreamer_sink_example_play ${apppkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_sink_example_play PUBLIC ${apppkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_sink_example_play PUBLIC ${apppkgs_CFLAGS_OTHER})

IF (INSTALL_EXAMPLE_APP)
	INSTALL(TARGETS nnstreamer_sink_example nnstreamer_sink_example_play RUNTIME DESTINATION ${EXAMPLE_EXEC_PREFIX})
ENDIF (INSTALL_EXAMPLE_APP)
