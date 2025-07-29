import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


    def generate(self, prompt, system_prompt):
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