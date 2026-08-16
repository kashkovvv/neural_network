add_executable(nn_cmake_smoke_test cmake_smoke_test.cpp)

target_link_libraries(nn_cmake_smoke_test PRIVATE nn_warnings nn_sanitizers)

add_test(NAME cmake.smoke COMMAND nn_cmake_smoke_test)

add_executable(nn_tensor_metadata_test tensor_metadata_test.cpp)

target_link_libraries(nn_tensor_metadata_test PRIVATE nn nn_warnings nn_sanitizers)

add_test(NAME tensor.metadata COMMAND nn_tensor_metadata_test)

add_executable(nn_tensor_access_test tensor_access_test.cpp)

target_link_libraries(nn_tensor_access_test PRIVATE nn nn_warnings nn_sanitizers)

add_test(NAME tensor.access COMMAND nn_tensor_access_test)
