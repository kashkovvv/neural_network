cmake_minimum_required(VERSION 3.20)

project(neural_network VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nn_warnings INTERFACE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    nn_warnings
    INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Werror
      -Wconversion
      -Wsign-conversion
      -Wshadow
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Woverloaded-virtual
      -Wnull-dereference
      -Wdouble-promotion
      -Wformat=2
      -Wimplicit-fallthrough
      -Wduplicated-cond
      -Wduplicated-branches
      -Wlogical-op
      -Wuseless-cast
  )
endif()

option(NN_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(nn_sanitizers INTERFACE)

if(NN_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    nn_sanitizers
    INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
  )

  target_link_options(nn_sanitizers INTERFACE -fsanitize=address,undefined)
endif()

add_library(nn INTERFACE)

target_include_directories(nn INTERFACE "${PROJECT_SOURCE_DIR}/include")

include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
