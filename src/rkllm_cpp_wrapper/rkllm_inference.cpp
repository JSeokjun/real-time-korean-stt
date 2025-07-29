#include <iostream>
#include <string>
#include <vector>
#include <cstring> // for strcpy

// C API 헤더 파일 포함
#include "rkllm.h"

// 콜백 함수: LLM의 응답을 실시간으로 출력합니다.
void llm_result_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
        fflush(stdout);
    } else if (state == RKLLM_RUN_FINISH) {
        // 완료
    } else if (state == RKLLM_RUN_ERROR) {
        fprintf(stderr, "LLM run error\n");
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "사용법: %s <rkllm 모델 디렉터리 경로>\n", argv[0]);
        return 1;
    }

    char* model_dir_path = argv[1];
    
    // Python 스크립트로부터 프롬프트를 읽어옵니다.
    std::string full_prompt;
    std::string line;
    while (std::getline(std::cin, line)) {
        full_prompt += line + "\n";
    }

    // 1. LLM 핸들 생성
    LLMHandle handle = nullptr;
    
    // 2. 기본 파라미터 생성 및 설정
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = model_dir_path;
    param.max_new_tokens = 512;
    // param.is_async = true; // 비동기 모드가 필요하면 활성화

    // 3. LLM 초기화
    int ret = rkllm_init(&handle, &param, llm_result_callback);
    if (ret != 0) {
        fprintf(stderr, "rkllm_init 실패, 오류 코드: %d\n", ret);
        return -1;
    }
    
    // 4. 입력 데이터 설정
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    rkllm_input.input_type = RKLLM_INPUT_PROMPT;
    rkllm_input.prompt_input = full_prompt.c_str();

    // 5. 추론 실행 (동기 방식)
    ret = rkllm_run(handle, &rkllm_input, nullptr, nullptr);
    if (ret != 0) {
        fprintf(stderr, "rkllm_run 실패, 오류 코드: %d\n", ret);
        rkllm_destroy(handle);
        return -1;
    }

    // 6. 자원 해제
    rkllm_destroy(handle);

    printf("\n");
    return 0;
} 