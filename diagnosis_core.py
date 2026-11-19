"""CSV validation and CPU inference for the supplied 600–24–24–7 checkpoint.

Architecture and StandardScaler preprocessing were confirmed from the original
training notebook. Fixed mean/scale were reconstructed from its training CSV;
all 600 float32 values match the notebook's printed statistics exactly. New
CSVs are never used to fit a scaler. The sample is not an independent test set.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

CLASS_NAMES = [
    "정상", "A상 윗상 스위치 고장", "A상 아랫상 스위치 고장",
    "B상 윗상 스위치 고장", "B상 아랫상 스위치 고장",
    "C상 윗상 스위치 고장", "C상 아랫상 스위치 고장",
]
SAMPLES_PER_PHASE = 200
INPUT_SIZE = 600
MODEL_ASSUMPTION = (
    "원본 학습 코드 확인: 600→24→24→7 FNN, ReLU, 학습 데이터의 고정 StandardScaler 적용. "
    "정규화 계수 600개는 원본 노트북 출력과 일치합니다."
)


def resource_path(relative: str) -> Path:
    """소스 실행 / PyInstaller onefile 실행에서 동일한 내장 파일 경로."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


@dataclass(frozen=True)
class Dataset:
    currents: np.ndarray  # [N,600], Ia[0:200], Ib[200:400], Ic[400:600]
    labels: np.ndarray | None  # 표시·평가 전용. 모델 입력에 포함하지 않습니다.
    path: Path
    columns: list[str]


def _numeric_row(row: list[str]) -> bool:
    try:
        for value in row:
            float(value.strip())
        return True
    except ValueError:
        return False


def _named_columns(header: list[str]) -> tuple[list[int], int | None]:
    normalized = [re.sub(r"[\s_-]", "", c).lower() for c in header]
    label_candidates = [i for i, name in enumerate(normalized)
                        if name in {"faultid", "fault", "label", "classid", "class"}]
    if len(label_candidates) > 1:
        raise ValueError("정답 열이 여러 개입니다. fault_id 열 하나만 사용해 주세요.")
    phases: dict[str, dict[int, int]] = {"a": {}, "b": {}, "c": {}}
    pattern = re.compile(r"^i([abc])[_\s.\-\[(]?(\d+)[\])]?$", re.IGNORECASE)
    for index, name in enumerate(header):
        match = pattern.fullmatch(name.strip())
        if not match:
            continue
        phase, sample = match.group(1).lower(), int(match.group(2))
        if sample in phases[phase]:
            raise ValueError(f"I{phase}의 샘플 번호 {sample}가 중복되었습니다.")
        phases[phase][sample] = index
    selected: list[int] = []
    origin: int | None = None
    for phase in "abc":
        numbers = sorted(phases[phase])
        if numbers == list(range(200)):
            current_origin = 0
        elif numbers == list(range(1, 201)):
            current_origin = 1
        else:
            raise ValueError(
                "헤더는 Ia_0…Ia_199, Ib_0…Ib_199, Ic_0…Ic_199 또는 "
                "각각 1…200 형식이어야 합니다. 각 상은 정확히 200개입니다. "
                "헤더가 없다면 전류 600열 뒤에 선택적으로 fault_id를 넣어주세요."
            )
        if origin is not None and current_origin != origin:
            raise ValueError("세 상의 샘플 시작 번호를 동일하게 맞춰 주세요.")
        origin = current_origin
        selected.extend(phases[phase][n] for n in numbers)
    return selected, label_candidates[0] if label_candidates else None


def load_csv(path: str | Path) -> Dataset:
    """헤더 없는 600/601열 또는 명시적 Ia/Ib/Ic 헤더가 있는 CSV를 읽습니다.

    헤더 없는 파일은 첫 행도 데이터로 보존합니다. fault_id는 입력 600개와
    분리하고, 행 길이·NaN/Inf·정답 범위를 검사해 잘못된 추론을 막습니다.
    """
    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = source.read_text(encoding="cp949")
    if not text.strip():
        raise ValueError("CSV 파일이 비어 있습니다.")
    first_line = next(line for line in text.splitlines() if line.strip())
    delimiter = max((",", ";", "\t"), key=first_line.count)
    rows = [(line_number, row) for line_number, row in
            enumerate(csv.reader(text.splitlines(), delimiter=delimiter), 1)
            if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("CSV에 데이터 행이 없습니다.")
    first = [v.strip() for v in rows[0][1]]
    width = len(first)
    has_header = not _numeric_row(first)
    if has_header:
        selected, label_index = _named_columns(first)
        columns = [first[i] for i in selected]
        data_rows = rows[1:]
    else:
        if width not in (600, 601):
            raise ValueError(f"헤더 없는 CSV는 600열 또는 601열이어야 합니다. 현재 {width}열입니다.")
        selected = list(range(600))
        label_index = 600 if width == 601 else None
        columns = [f"I{phase}_{sample}" for phase in "abc" for sample in range(200)]
        data_rows = rows
    if not data_rows:
        raise ValueError("헤더 뒤에 데이터 행이 없습니다.")
    values = np.empty((len(data_rows), INPUT_SIZE), dtype=np.float32)
    labels = np.empty(len(data_rows), dtype=np.int64) if label_index is not None else None
    for output_index, (line_number, row) in enumerate(data_rows):
        if len(row) != width:
            raise ValueError(f"CSV {line_number}행: 열 수가 {len(row)}개입니다. 예상 {width}개.")
        try:
            parsed = np.array([float(row[i]) for i in selected], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"CSV {line_number}행: 전류에 숫자가 아닌 값 또는 빈칸이 있습니다.") from exc
        if not np.isfinite(parsed).all() or np.any(np.abs(parsed) > np.finfo(np.float32).max):
            raise ValueError(f"CSV {line_number}행: 전류에 NaN, Inf 또는 float32 범위 초과 값이 있습니다.")
        values[output_index] = parsed
        if label_index is not None:
            try:
                label = float(row[label_index])
            except ValueError as exc:
                raise ValueError(f"CSV {line_number}행: fault_id가 숫자가 아닙니다.") from exc
            if not np.isfinite(label) or label != int(label) or not 0 <= label <= 6:
                raise ValueError(f"CSV {line_number}행: fault_id는 0~6의 정수여야 합니다.")
            labels[output_index] = int(label)
    return Dataset(np.ascontiguousarray(values), labels, source, columns)


class FaultFNN(nn.Module):
    """학습 구조 변경 시 수정할 지점. 제공된 가중치의 키를 그대로 유지합니다."""
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(600, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 7),  # logits; softmax는 추론 결과에 한 번만 적용
        )

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        return self.model(currents)


class DiagnosisEngine:
    """신뢰할 수 있는 state_dict만 로드하는 CPU 추론기."""
    def __init__(self, model_path: str | Path, scaler_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path).resolve()
        self.scaler_path = Path(scaler_path) if scaler_path else self.model_path.with_name("scaler.npz")
        with np.load(self.scaler_path, allow_pickle=False) as scaler:
            self.mean = np.asarray(scaler["mean"], dtype=np.float64).copy()
            self.scale = np.asarray(scaler["scale"], dtype=np.float64).copy()
        if self.mean.shape != (600,) or self.scale.shape != (600,):
            raise ValueError("고정 정규화 계수는 각 600개이어야 합니다.")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("고정 정규화 계수가 올바르지 않습니다.")
        state = torch.load(self.model_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("이 프로그램은 제공된 FNN state_dict 형식의 .pt를 사용합니다.")
        self.model = FaultFNN()
        expected = self.model.state_dict()
        if set(state) != set(expected):
            raise ValueError("모델 가중치 키가 600→24→24→7 FNN 구조와 맞지 않습니다.")
        for key, target in expected.items():
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
                raise ValueError(f"모델 가중치 {key}의 크기가 예상 {tuple(target.shape)}와 다릅니다.")
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"모델 가중치 {key}에 올바르지 않은 값이 있습니다.")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.model_summary = "CPU · FNN 600 → 24 → 24 → 7 · StandardScaler · ReLU · Softmax"
        self.reconstruction_note = MODEL_ASSUMPTION

    def predict(self, currents: np.ndarray) -> np.ndarray:
        array = np.asarray(currents)
        if array.ndim != 2 or array.shape[1] != INPUT_SIZE:
            raise ValueError("모델 입력 크기는 [행 개수, 600]이어야 합니다.")
        if len(array) == 0:
            return np.empty((0, 7), dtype=np.float32)
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("모델 입력에 NaN 또는 Inf가 있습니다.")
        # StandardScaler의 float32 transform과 동일하게 빼기/나누기를 각각
        # float32에 기록합니다. 원본 파형은 변경하지 않으며 새 파일로 fit하지 않습니다.
        normalized = array.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            normalized -= self.mean
            normalized /= self.scale
        if not np.isfinite(normalized).all():
            raise ValueError("정규화된 입력에 NaN 또는 Inf가 있습니다. 입력 범위를 확인하세요.")
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(array), 256):
                # fault_id는 이 함수로 전달되지 않습니다.
                logits = self.model(torch.from_numpy(normalized[start:start + 256]))
                probability = torch.softmax(logits, dim=1)
                if not bool(torch.isfinite(probability).all()):
                    raise ValueError("모델 출력에 NaN 또는 Inf가 있습니다. 입력 범위를 확인하세요.")
                chunks.append(probability.cpu().numpy())
        return np.concatenate(chunks, axis=0)
