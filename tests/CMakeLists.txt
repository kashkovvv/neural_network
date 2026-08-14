add_executable(nn_cmake_smoke_test cmake_smoke_test.cpp)

target_link_libraries(nn_cmake_smoke_test PRIVATE nn_warnings nn_sanitizers)

add_test(NAME cmake.smoke COMMAND nn_cmake_smoke_test)
