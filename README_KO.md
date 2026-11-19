# SEED Fault Diagnosis — CMD용 간소화 소스

이 폴더는 이미 학습된 `.pt` 모델로 전류 파형을 진단하고, GUI 프로그램을 EXE로 만드는 소스입니다. 새 모델을 학습하는 코드는 포함하지 않습니다.

## 1. CMD는 어디서 여나요?

압축을 해제하고 `main.py`가 보이는 폴더를 탐색기로 엽니다. 탐색기 주소창에 `cmd`를 입력하고 Enter를 누르면 그 폴더에서 명령 프롬프트가 열립니다. 아래 명령어는 CMD 기준입니다.

예를 들어 `C:\SEED_FaultDiagnosis_Simple`에 해제했다면 다음과 같이 이동할 수도 있습니다.

```bat
cd /d "C:\SEED_FaultDiagnosis_Simple"
```

`cd`는 폴더 이동, `/d`는 드라이브도 함께 변경, 따옴표는 경로의 공백을 처리하기 위한 것입니다.

## 2. 처음 한 번: 실행·빌드 환경 준비

Windows 64비트 Python 3.14에서 기존 배포본을 빌드했습니다. 이 간소화본은 같은 라이브러리 버전을 사용합니다. 최초 설치에는 인터넷이 필요합니다.

```bat
python --version
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

각 명령이 끝난 뒤 다음 명령을 입력합니다. 설치 중 오류가 발생했다면 오류를 해결한 다음 진행합니다. `python`이 인식되지 않으면 Python 설치와 PATH 설정을 확인합니다.

`.venv`는 이 프로젝트의 Python과 라이브러리를 모아 두는 개발 환경입니다. `activate.bat`는 CMD에서 이 환경을 선택합니다. 활성화되면 보통 명령줄 앞에 `(.venv)`가 표시됩니다. 다른 PC에서는 `.venv`를 복사하지 말고 위 명령으로 새로 만듭니다.

이미 필요한 라이브러리가 설치된 Python을 사용한다면 환경 준비를 생략하고 바로 실행할 수도 있습니다.

## 3. GUI를 실행해 확인

```bat
python main.py
```

예제 CSV를 열고 모델이 진단합니다. Ia·Ib·Ic 원본 파형, 예측 고장, 실제 고장 ID, 7개 softmax 값을 확인할 수 있습니다. 이 명령은 모델 학습이나 EXE 생성이 아닙니다.

같은 CMD에서 다음 명령을 입력하려면 먼저 GUI 창을 닫습니다.

## 4. EXE 만들기

```bat
python build_exe.py
```

완료 후 `dist\SEED_FaultDiagnosis.exe`가 만들어집니다. 모델·scaler·예제 CSV와 필요한 런타임 라이브러리가 포함된 단일 실행 파일입니다. 실행할 PC에는 별도의 Python 설치가 필요하지 않습니다. 첫 실행에는 압축을 푸는 시간이 걸릴 수 있습니다.

소스나 모델 파일을 수정한 뒤에는 이 명령을 다시 실행해야 새 EXE에 반영됩니다. 소스만 수정해도 기존 EXE가 자동으로 바뀌는 것은 아닙니다.

다음 날 새 CMD를 열었다면 해당 폴더로 이동한 후 `.venv\Scripts\activate.bat`를 먼저 실행합니다. 패키지 설치는 환경이나 요구 버전을 바꾸지 않는 한 매번 반복하지 않습니다.

## 5. 폴더 구성 — 총 11개 파일

| 경로 | 역할 |
| --- | --- |
| `main.py` | 프로그램 시작 |
| `gui.py` | 화면, 파형, 버튼, 결과 표시 |
| `diagnosis_core.py` | CSV 읽기, 신경망 구조, 전처리, 추론 |
| `build_exe.py` | EXE 생성 명령의 실행 파일 |
| `SEED_FaultDiagnosis.spec` | EXE에 포함할 파일과 라이브러리 설정 |
| `requirements.txt` | 실행·빌드 라이브러리 설치 목록 하나로 통합 |
| `README_KO.md` | 이 설명서 |
| `LICENSE_THIRD_PARTY.txt` | 사용한 외부 라이브러리 안내 |
| `assets/fnn_fault_model.pt` | 학습된 모델 가중치 |
| `assets/scaler.npz` | 학습 시 사용한 고정 평균·스케일 |
| `assets/sample.csv` | GUI가 시작할 때 여는 예제 데이터 |

수정할 때는 주로 `gui.py`와 `diagnosis_core.py`를 봅니다. 나머지는 실행 또는 빌드를 지원합니다. 파일 이름은 강의자료와 대응되도록 유지했습니다.

`sample.csv`는 현재 코드가 시작할 때 자동으로 여는 파일입니다. 단순히 삭제하면 시작할 때 오류가 나므로 포함합니다. `.spec`도 빌드 코드가 직접 참조하므로 유지합니다.

## 6. 입력과 모델

CSV 한 행은 `[Ia 200개, Ib 200개, Ic 200개, 선택적 fault_id]`입니다. 정답 열을 제외한 600개 전류만 신경망 입력으로 사용합니다.

모델은 `600 → 24 → 24 → 7` FNN이며, 은닉층 활성화는 ReLU입니다. 고정 표준화를 적용한 뒤 추론하고 마지막 출력에 softmax를 적용합니다. 파형에는 표준화 전 원본 값을 표시합니다.

| fault ID | 의미 |
| --- | --- |
| 0 | 정상 |
| 1 / 2 | A상 윗상 / 아랫상 스위치 고장 |
| 3 / 4 | B상 윗상 / 아랫상 스위치 고장 |
| 5 / 6 | C상 윗상 / 아랫상 스위치 고장 |

새 CSV를 열 때 scaler를 다시 학습하지 않습니다. 새로운 모델을 적용하려면 그 모델과 함께 학습에 사용한 scaler를 교체해야 합니다. 신경망 구조나 활성화 함수가 바뀌면 `diagnosis_core.py`의 모델 정의도 맞춰야 합니다. 새 모델 학습 자체는 별도의 학습 코드/노트북에서 수행합니다.

동봉 CSV는 독립 시험 데이터가 아닙니다. 예제 정답 일치율은 동작 확인용이며, 현장 일반화 정확도나 보정된 고장 발생 확률을 의미하지 않습니다.

## 7. 간소화하면서 제외한 파일

| 기존 파일 | 제외 이유 |
| --- | --- |
| `tests/`, `model_validation.json`, `exe_build_metadata.json` | 개발 검증용 테스트·기록. GUI 실행과 EXE 생성에는 필요 없음 |
| `preview.png` | 미리보기 이미지 |
| `build_exe.bat`, `run_gui.bat` | CMD에서 Python 명령으로 실행하므로 생략 |
| `requirements-build.txt` | 설치 목록을 `requirements.txt`로 통합 |

이 목록은 새 간소화본의 구성입니다. 원본 소스 폴더의 파일을 직접 삭제하지 않았습니다. 기존 폴더에서 `requirements-build.txt`만 삭제하면 기존 BAT의 설치 과정이 깨지므로, 간소화본을 새 폴더에 풀어 사용합니다.

실행·빌드 중 생기는 `__pycache__`와 `build`는 재생성 가능한 임시 파일입니다. `.venv`를 삭제하면 개발 환경을 다시 설치해야 합니다. `dist`는 완성된 EXE가 있는 결과 폴더이므로 배포할 파일을 보관합니다. 실행·빌드가 진행 중일 때는 관련 폴더를 삭제하지 않습니다.

## 참고

- Python 가상환경: https://docs.python.org/3/library/venv.html
- PyInstaller 사용: https://pyinstaller.org/en/stable/usage.html

간소화본은 원본 GUI·추론 코드·모델·scaler·예제 CSV·패키징 설정을 그대로 유지하며, 설치 목록과 안내 문구를 정리했습니다.
