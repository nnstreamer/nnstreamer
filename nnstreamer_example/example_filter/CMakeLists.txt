CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# demo app
ADD_EXECUTABLE(nnstreamer_example_filter nnstreamer_example_filter.c)

TARGET_LINK_LIBRARIES(nnstreamer_example_filter ${pkgs_LIBRARIES})
TARGET_INCLUDE_DIRECTORIES(nnstreamer_example_filter PUBLIC ${pkgs_INCLUDE_DIRS})
TARGET_COMPILE_OPTIONS(nnstreamer_example_filter PUBLIC ${pkgs_CFLAGS_OTHER})

# do not install example app
# INSTALL(TARGETS nnstreamer_example_filter RUNTIME DESTINATION ${EXEC_PREFIX})
