cmake_minimum_required(VERSION 3.10)
project(rkllm_inference_project CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python 스크립트에서 -D 옵션으로 이 경로를 전달합니다.
if(NOT DEFINED RKLLM_RUNTIME_DIR OR NOT RKLLM_RUNTIME_DIR)
    message(FATAL_ERROR "RKLLM_RUNTIME_DIR path must be passed via -D option and cannot be empty.")
endif()

message(STATUS "Passed RKLLM_RUNTIME_DIR: ${RKLLM_RUNTIME_DIR}")

# 헤더 및 라이브러리 경로 설정
set(RKLLM_INCLUDE_DIR "${RKLLM_RUNTIME_DIR}/librkllm_api/include")
set(RKLLM_LIB_DIR "${RKLLM_RUNTIME_DIR}/librkllm_api/aarch64")

# 경로 존재 여부 확인
if(NOT IS_DIRECTORY ${RKLLM_INCLUDE_DIR})
    message(FATAL_ERROR "RKLLM include directory NOT FOUND at: ${RKLLM_INCLUDE_DIR}")
endif()
if(NOT IS_DIRECTORY ${RKLLM_LIB_DIR})
    message(FATAL_ERROR "RKLLM library directory NOT FOUND at: ${RKLLM_LIB_DIR}")
endif()

message(STATUS "Found RKLLM include directory: ${RKLLM_INCLUDE_DIR}")
message(STATUS "Found RKLLM library directory: ${RKLLM_LIB_DIR}")

# 실행 파일 생성
add_executable(rkllm_inference rkllm_inference.cpp)

# 헤더 파일 경로 추가
target_include_directories(rkllm_inference PUBLIC ${RKLLM_INCLUDE_DIR})

# 라이브러리 경로 추가
target_link_directories(rkllm_inference PUBLIC ${RKLLM_LIB_DIR})

# 라이브러리 링크
target_link_libraries(rkllm_inference rkllmrt)

# C++11 지원을 위한 플래그
set_target_properties(rkllm_inference PROPERTIES CXX_STANDARD 11) 