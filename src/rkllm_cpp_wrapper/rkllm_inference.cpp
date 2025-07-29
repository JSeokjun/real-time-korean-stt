#include <iostream>
#include <string>
#include <vector>
#include "rkllm.h" // RKLLM C++ API 헤더

// RKLLM에서 생성되는 토큰을 실시간으로 출력하기 위한 콜백 함수
void llm_callback(const char* token) {
    std::cout << token << std::flush;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "사용법: " << argv[0] << " <rkllm 모델 디렉터리 경로>" << std::endl;
        return 1;
    }

    std::string model_dir_path = argv[1];
    
    // Python 스크립트로부터 표준 입력(stdin)을 통해 전체 프롬프트를 읽어옵니다.
    std::string full_prompt;
    std::string line;
    while (std::getline(std::cin, line)) {
        full_prompt += line + "\n";
    }

    // RKLLM 초기화
    RKLLM llm;
    int ret = llm.load_llm(model_dir_path.c_str());
    if (ret != 0) {
        std::cerr << "RKLLM 모델 로드 실패, 오류 코드: " << ret << std::endl;
        return -1;
    }

    // 생성(generation) 설정
    RKLLM_GEN_CONFIG gen_config;
    gen_config.max_new_tokens = 512; // 최대 생성 토큰 수
    gen_config.stream = true;        // 스트리밍 출력 활성화

    // 프롬프트를 기반으로 텍스트 생성 시작
    ret = llm.generate(full_prompt, gen_config, llm_callback);
    if (ret != 0) {
        std::cerr << "RKLLM 생성 실패, 오류 코드: " << ret << std::endl;
        return -1;
    }
    
    std::cout << std::endl;

    return 0;
} 