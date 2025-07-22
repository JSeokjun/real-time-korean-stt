import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM\

try:
    from rkllm.api import RKLLM

except ImportError:
    print("경고: rkllm 모듈을 찾을 수 없습니다. LLM 기능을 사용할 수 없습니다.")
    print("Rockchip 장치에서 RKLLM 런타임이 설치된 상태로 실행하고 있는지 확인하십시오.")
    RKLLM = None

class GemmaCPU:
    """
    CPU 또는 MPS 장치에서 Hugging Face Transformers를 사용하여 Gemma 모델을 관리합니다.
    """
    def __init__(self, model_dir_path, device="cpu"):
        """
        토크나이저와 모델을 로드합니다.

        Args:
            model_dir_path (str): 모델 파일이 포함된 디렉토리의 경로입니다.
            device (str): 모델을 로드할 장치입니다. ("cpu" 또는 "mps")
        """
        print(f"\n--- Hugging Face 모델 초기화 중 ({device}) ---")
        if not os.path.isdir(model_dir_path):
            raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            torch_dtype=torch.float16, # mps는 float16, cpu는 bfloat16이 좋으나 호환성을 위해 float16
            low_cpu_mem_usage=True
        ).to(device)
        self.device = device
        print(f"Hugging Face Gemma 모델을 성공적으로 로드했습니다. ({device})")


    def generate(self, prompt, system_prompt="당신은 주어진 텍스트를 분석하고 간결한 요약이나 통찰력을 제공하는 유용한 어시스턴트입니다."):
        """
        주어진 프롬프트를 기반으로 텍스트를 생성합니다.

        Args:
            prompt (str): 사용자의 입력 텍스트입니다.
            system_prompt (str, optional): 모델의 역할을 정의하는 시스템 프롬프트입니다.

        Returns:
            str: 모델이 생성한 응답입니다.
        """
        print(f"\n[LLM] 프롬프트에 대한 응답 생성 중: \"{prompt[:50]}...\"")

        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
        ]
        
        # chat_template을 사용하여 프롬프트 형식 지정
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()


class GemmaRK3588:
    """
    Rockchip NPU에서 RKLLM 모델을 사용하여 텍스트 생성을 관리합니다.
    """
    def __init__(self, model_dir_path, rkllm_lib_path):
        """
        RKLLM 런타임을 초기화하고 모델을 로드합니다.

        Args:
            model_dir_path (str): .rkllm 모델과 토크나이저 파일이 포함된 디렉토리의 경로입니다.
            rkllm_lib_path (str): 'librkllmrt.so' 파일의 경로입니다.
        """
        if RKLLM is None:
            raise ImportError("RKLLM을 사용할 수 없습니다. GemmaRK3588을 초기화할 수 없습니다.")

        if not os.path.isdir(model_dir_path):
            raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir_path}")
        
        if not os.path.exists(rkllm_lib_path):
            raise FileNotFoundError(f"RKLLM 라이브러리를 찾을 수 없습니다: {rkllm_lib_path}")

        print("\n--- RKLLM 초기화 중 ---")
        # RKLLM 생성자는 라이브러리 경로를 직접 인자로 받을 수 있습니다.
        self.llm = RKLLM(config={'librkllmrt_path': rkllm_lib_path})
        
        # 모델 디렉토리에서 .rkllm 파일 찾기
        rkllm_file = None
        for file in os.listdir(model_dir_path):
            if file.endswith('.rkllm'):
                rkllm_file = os.path.join(model_dir_path, file)
                break

        if not rkllm_file:
            raise FileNotFoundError(f"디렉토리에서 .rkllm 파일을 찾을 수 없습니다: {model_dir_path}")

        print(f"RKLLM 모델 파일을 찾았습니다: {os.path.basename(rkllm_file)}")
        print(f"디렉토리에서 모델을 로드하는 중: {model_dir_path}")

        ret = self.llm.load_llm(llm_path=rkllm_file,
                                model_path=model_dir_path)
        if ret != 0:
            raise RuntimeError(f"RKLLM 모델 로드에 실패했습니다, 오류 코드: {ret}")

        print("Gemma LLM 모델을 성공적으로 로드했습니다.")

    def generate(self, prompt, system_prompt="당신은 주어진 텍스트를 분석하고 간결한 요약이나 통찰력을 제공하는 유용한 어시스턴트입니다."):
        """
        주어진 프롬프트를 기반으로 텍스트를 생성합니다.

        Args:
            prompt (str): 사용자의 입력 텍스트입니다.
            system_prompt (str, optional): 모델의 역할을 정의하는 시스템 프롬프트입니다.

        Returns:
            str: 모델이 생성한 응답입니다.
        """
        print(f"\n[LLM] 프롬프트에 대한 응답 생성 중: \"{prompt[:50]}...\"")

        output = self.llm.generate(prompt,
                                   system_prompt=system_prompt,
                                   max_new_tokens=512)
        
        return output 