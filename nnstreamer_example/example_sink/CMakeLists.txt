CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# example 1
ADD_EXECUTABLE(nnstreamer_sink_example nnstreamer_sink_example.c)

TARGET_LINK_LIBRARIES(nnstreamer_sink_example ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_sink_example PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_sink_example PUBLIC ${pkgs_CFLAGS_OTHER})

# example 2
ADD_EXECUTABLE(nnstreamer_sink_example_play nnstreamer_sink_example_play.c)

TARGET_LINK_LIBRARIES(nnstreamer_sink_example_play gstapp-1.0 ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_sink_example_play PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_sink_example_play PUBLIC ${pkgs_CFLAGS_OTHER})

# do not install sample app
# INSTALL(TARGETS nnstreamer_sink_example nnstreamer_sink_example_play RUNTIME DESTINATION ${EXEC_PREFIX})
