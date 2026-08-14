add_executable(nn_cmake_smoke_test cmake_smoke_test.cpp)

add_test(NAME cmake.smoke COMMAND nn_cmake_smoke_test)
