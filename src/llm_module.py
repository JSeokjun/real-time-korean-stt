import os
import torch
import subprocess
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


class GemmaRK3588:
    """
    Rockchip NPU에서 C++ 바이너리를 통해 RKLLM 모델을 사용하여 텍스트 생성을 관리합니다.
    """
    def __init__(self, model_dir_path, rkllm_lib_path, **kwargs):
        """
        RKLLM C++ 바이너리 실행을 위한 경로를 설정하고 필요시 컴파일합니다.

        Args:
            model_dir_path (str): .rkllm 모델과 토크나이저 파일이 포함된 디렉토리의 경로입니다.
            rkllm_lib_path (str): 'librkllmrt.so' 파일의 경로입니다.
        """
        if not os.path.isdir(model_dir_path):
            raise FileNotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {model_dir_path}")

        if not os.path.exists(rkllm_lib_path):
            raise FileNotFoundError(f"RKLLM 라이브러리 파일을 찾을 수 없습니다: {rkllm_lib_path}")

        self.model_dir_path = model_dir_path
        self.rkllm_lib_path = rkllm_lib_path
        
        # C++ 래퍼 관련 경로 설정
        project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.wrapper_dir = os.path.join(project_root, "src/rkllm_cpp_wrapper")
        self.build_dir = os.path.join(self.wrapper_dir, "build")
        self.inference_binary = os.path.join(self.build_dir, "rkllm_inference")
        
        print(f"\n--- RKLLM C++ 래퍼 초기화 ---")
        print(f"모델 디렉토리: {self.model_dir_path}")
        print(f"추론 바이너리 경로: {self.inference_binary}")

        # 바이너리가 없으면 컴파일 시도
        if not os.path.exists(self.inference_binary):
            print("추론 바이너리를 찾을 수 없습니다. 컴파일을 시작합니다...")
            self._build_inference_app()

    def _build_inference_app(self):
        """CMake를 사용하여 C++ 추론 애플리케이션을 빌드합니다."""
        # 이전 빌드가 문제를 일으키지 않도록 build 디렉터리 정리
        if os.path.exists(self.build_dir):
            import shutil
            shutil.rmtree(self.build_dir)
        
        os.makedirs(self.build_dir)
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        rkllm_runtime_dir = os.path.join(project_root, "rknn-llm/rkllm-runtime/Linux")

        if not os.path.isdir(rkllm_runtime_dir):
             raise FileNotFoundError(f"RKLLM 런타임 디렉토리를 찾을 수 없습니다: {rkllm_runtime_dir}")

        cmake_cmd = [
            "cmake",
            f"-DRKLLM_RUNTIME_DIR={rkllm_runtime_dir}",
            ".."
        ]
        make_cmd = ["make", "-j4"]

        try:
            print(f"CMake 실행: {' '.join(cmake_cmd)}")
            # stderr=subprocess.STDOUT 옵션으로 모든 출력을 하나로 합쳐서 확인
            cmake_result = subprocess.run(cmake_cmd, cwd=self.build_dir, check=True, capture_output=True, text=True)
            print("--- CMake Output ---")
            print(cmake_result.stdout)
            if cmake_result.stderr:
                print("--- CMake Error ---")
                print(cmake_result.stderr)
            
            print(f"Make 실행: {' '.join(make_cmd)}")
            make_result = subprocess.run(make_cmd, cwd=self.build_dir, check=True, capture_output=True, text=True)
            print("--- Make Output ---")
            print(make_result.stdout)
            if make_result.stderr:
                print("--- Make Error ---")
                print(make_result.stderr)

            print("컴파일 성공!")
        except FileNotFoundError:
            raise RuntimeError("CMake 또는 Make를 찾을 수 없습니다. 'sudo apt install cmake build-essential'을 실행하여 설치해주세요.")
        except subprocess.CalledProcessError as e:
            print("컴파일 실패!")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise RuntimeError(f"C++ 추론 앱 빌드에 실패했습니다. 오류: {e}")

    def generate(self, prompt, system_prompt):
        """
        컴파일된 C++ 바이너리를 실행하여 텍스트를 생성합니다.

        Args:
            prompt (str): 사용자의 입력 텍스트입니다.
            system_prompt (str): 모델의 역할을 정의하는 시스템 프롬프트입니다.

        Returns:
            str: 모델이 생성한 응답입니다.
        """
        print(f"\n[LLM] C++ 바이너리로 응답 생성 중: \"{prompt[:50]}...\"")
        
        full_prompt = f"{system_prompt}\n\n[원문]\n{prompt}"

        try:
            # 런타임에 라이브러리를 찾을 수 있도록 환경 변수 설정
            env = os.environ.copy()
            lib_dir = os.path.dirname(self.rkllm_lib_path)
            env['LD_LIBRARY_PATH'] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

            process = subprocess.Popen(
                [self.inference_binary, self.model_dir_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                env=env # 수정된 환경 변수 적용
            )

            # C++ 프로그램에 프롬프트 전달 및 결과 수신
            stdout, stderr = process.communicate(input=full_prompt, timeout=60) # 60초 타임아웃
            
            if process.returncode != 0:
                print(f"C++ 바이너리 실행 오류 (코드: {process.returncode}):")
                print(stderr)
                return f"[오류] 추론 중 문제가 발생했습니다: {stderr}"
            
            return stdout.strip()

        except FileNotFoundError:
            return "[오류] 추론 바이너리를 찾을 수 없습니다. 빌드가 제대로 되었는지 확인하세요."
        except subprocess.TimeoutExpired:
            process.kill()
            return "[오류] 추론 시간이 초과되었습니다."
        except Exception as e:
            return f"[오류] 추론 중 예외가 발생했습니다: {str(e)}" 