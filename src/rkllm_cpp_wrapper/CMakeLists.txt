cmake_minimum_required(VERSION 3.10)
project(rkllm_inference_project CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RKLLM 라이브러리 및 헤더 파일 경로 설정
# 이 경로는 Python 스크립트에서 동적으로 전달됩니다.
set(RKLLM_RUNTIME_DIR "" CACHE PATH "Path to rknn-llm/rkllm-runtime/Linux directory")

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