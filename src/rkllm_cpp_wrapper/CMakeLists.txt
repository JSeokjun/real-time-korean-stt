cmake_minimum_required(VERSION 3.10)
project(rkllm_inference_project CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKLLM 라이브러리 및 헤더 파일 경로 설정
# Python 스크립트에서 -D 옵션으로 이 경로를 전달합니다.
if(NOT RKLLM_RUNTIME_DIR)
    message(FATAL_ERROR "RKLLM_RUNTIME_DIR path must be passed via -D option.")
endif()

message(STATUS "Using RKLLM_RUNTIME_DIR: ${RKLLM_RUNTIME_DIR}")

# 헤더 파일 경로 추가
include_directories(${RKLLM_RUNTIME_DIR}/librkllm_api/include)

# 라이브러리 경로 추가
link_directories(${RKLLM_RUNTIME_DIR}/librkllm_api/aarch64)

# 실행 파일 생성
add_executable(rkllm_inference rkllm_inference.cpp)

# 라이브러리 링크
target_link_libraries(rkllm_inference rkllmrt)

# C++11 지원을 위한 플래그
set_target_properties(rkllm_inference PROPERTIES CXX_STANDARD 11) 