"""PyQt6 user interface for the SEED Lab fault diagnosis application.

The neural-network implementation lives in diagnosis_core.py. This file only
loads datasets, requests inference, and presents original currents and results.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QLayout, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from diagnosis_core import CLASS_NAMES, load_csv, resource_path


STYLE = """
QMainWindow, QWidget#root { background: #f1f4f8; color: #18283e; }
QWidget { font-family: 'Malgun Gothic'; font-size: 12px; }
QFrame#card { background: white; border: 1px solid #dce3ec; border-radius: 9px; }
QLabel { border: none; background: transparent; }
QLabel#brand { font-size: 25px; font-weight: 700; color: #132a46; }
QLabel#section { font-size: 15px; font-weight: 700; color: #132a46; }
QLabel#muted { color: #62738a; }
QLabel#prediction { font-size: 21px; font-weight: 700; color: #123d68; }
QPushButton { min-height: 30px; padding: 4px 13px; border: 1px solid #c6d1df;
              border-radius: 5px; background: white; color: #213c5c; }
QPushButton:hover { background: #edf3fa; border-color: #7a99bb; }
QPushButton:disabled { color: #9ca7b5; background: #f5f6f8; border-color: #dde2e8; }
QPushButton#primary { background: #163d66; color: white; border-color: #163d66; }
QPushButton#primary:hover { background: #24527e; }
QPushButton#primary:disabled { background: #9daebe; border-color: #9daebe; }
QComboBox, QSpinBox { padding: 5px; min-height: 25px; background: white;
                    border: 1px solid #ccd6e2; border-radius: 4px; }
QTableWidget { background: white; alternate-background-color: #f5f8fb;
               gridline-color: #e8edf3; border: none; selection-background-color: #dfeaf6;
               selection-color: #153757; }
QHeaderView::section { background: #eef3f8; color: #435a74; border: none;
                       border-bottom: 1px solid #dce3ec; padding: 7px; font-weight: 600; }
QProgressBar { background: #edf2f7; border: none; border-radius: 4px; min-height: 12px;
               text-align: center; }
QProgressBar::chunk { background: #5686b5; border-radius: 4px; }
QStatusBar { background: #e8eef5; color: #4a607b; }
QSplitter::handle { background: #f1f4f8; }
"""


class InferenceWorker(QThread):
    """CSV 읽기와 CPU 추론을 한 작업으로 직렬화합니다."""

    progress = pyqtSignal(int, int)
    result = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, engine, path: Path, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.path = path

    def run(self):
        try:
            dataset = load_csv(self.path)
            count = len(dataset.currents)
            if count == 0:
                raise ValueError("CSV에 진단할 데이터 행이 없습니다.")
            output = np.empty((count, 7), dtype=np.float32)
            for start in range(0, count, 128):
                if self.isInterruptionRequested():
                    return
                stop = min(start + 128, count)
                # fault_id는 모델에 전달하지 않습니다. 입력은 전류 600개뿐입니다.
                values = np.asarray(self.engine.predict(dataset.currents[start:stop]))
                if values.shape != (stop - start, 7):
                    raise ValueError(f"모델 출력 형상이 올바르지 않습니다: {values.shape}")
                if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise ValueError("모델이 유효하지 않은 확률을 반환했습니다.")
                if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
                    raise ValueError("모델 출력 확률의 합이 1이 아닙니다.")
                output[start:stop] = values
                self.progress.emit(stop, count)
            if not self.isInterruptionRequested():
                self.result.emit(dataset, output)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class WaveformCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(8, 3.6), dpi=100, facecolor="white")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.085, right=0.98, bottom=0.25, top=0.91)
        self.lines = []
        colors = ("#265fa1", "#bc722e", "#327a66")
        styles = ("-", "--", "-.")
        for name, color, style in zip(("Ia", "Ib", "Ic"), colors, styles):
            line, = self.axes.plot([], [], label=name, color=color, linestyle=style, linewidth=1.65)
            self.lines.append(line)
        self.axes.set_xlabel("Sample index", fontfamily="Times New Roman", fontsize=12)
        self.axes.set_ylabel("Current (CSV units)", fontfamily="Times New Roman", fontsize=12)
        self.axes.set_xlim(0, 199)
        self.axes.set_ylim(-1, 1)
        self.axes.grid(True, alpha=0.25, linewidth=0.6)
        self.axes.legend(loc="upper right", ncol=3, framealpha=0.95)
        self.axes.tick_params(labelsize=10)
        for spine in self.axes.spines.values():
            spine.set_color("#bcc8d6")

    def show_currents(self, currents):
        values = np.asarray(currents).reshape(3, 200)
        for line, phase in zip(self.lines, values):
            line.set_data(np.arange(200), phase)
        self.axes.set_xlim(0, 199)
        low, high = float(values.min()), float(values.max())
        margin = max((high - low) * 0.12, abs(high) * 0.05, 1e-6)
        self.axes.set_ylim(low - margin, high + margin)
        self.draw_idle()

    def clear_waveforms(self):
        for line in self.lines:
            line.set_data([], [])
        self.draw_idle()


def card():
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(9)
    return frame, layout


def table_item(value, centered=False):
    item = QTableWidgetItem(str(value))
    if centered:
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


class MainWindow(QMainWindow):
    """CSV 행 탐색, 파형 표시, softmax 결과 및 결과 내보내기."""

    def __init__(self, engine, auto_load_sample=True):
        super().__init__()
        # 일부 offscreen 환경에서도 Windows 기본 한글 글꼴을 직접 등록합니다.
        if "Malgun Gothic" not in QFontDatabase.families() and os.name == "nt":
            font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
            if font_file.is_file():
                QFontDatabase.addApplicationFont(str(font_file))
        self.setFont(QFont("Malgun Gothic", 9))
        self.engine = engine
        self.dataset = None
        self.probabilities = None
        self.visible_rows = np.array([], dtype=int)
        self.current_row = -1
        self.worker = None
        self.last_error = ""
        self._updating = False
        self.setWindowTitle("SEED Lab | 3상 전류 고장진단")
        self.resize(1440, 940)
        self.setMinimumSize(900, 600)
        screen = QApplication.primaryScreen()
        if screen is not None and QApplication.platformName() != "offscreen":
            available = screen.availableGeometry()
            self.resize(min(1440, available.width()), min(940, available.height() - 40))
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._set_busy(False)
        if auto_load_sample:
            QTimer.singleShot(0, self.open_sample)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        # 작은 해상도/큰 DPI에서는 내용 크기를 유지하고 스크롤로 모두 접근합니다.
        root.setMinimumSize(1100, 918)
        root.setFixedHeight(918)
        root.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        layout = QVBoxLayout(root)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(22, 18, 22, 14)
        layout.setSpacing(12)

        heading = QHBoxLayout()
        title = QLabel("SEED  /  3상 전류 고장진단")
        title.setObjectName("brand")
        heading.addWidget(title)
        heading.addStretch()
        subtitle = QLabel("Ia · Ib · Ic  /  200 samples per phase  /  7 classes")
        subtitle.setObjectName("muted")
        heading.addWidget(subtitle)
        layout.addLayout(heading)

        toolbar = QHBoxLayout()
        self.open_button = QPushButton("CSV 열기")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_csv)
        self.sample_button = QPushButton("내장 예제 열기")
        self.sample_button.clicked.connect(self.open_sample)
        self.rerun_button = QPushButton("전체 다시 진단")
        self.rerun_button.clicked.connect(self.rerun)
        self.export_button = QPushButton("진단 결과 CSV 저장")
        self.export_button.clicked.connect(self.export_results)
        for button in (self.open_button, self.sample_button, self.rerun_button, self.export_button):
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(240)
        self.progress_bar.setFormat("%p%")
        toolbar.addWidget(self.progress_bar)
        layout.addLayout(toolbar)

        self.file_label = QLabel("CSV를 열어 진단을 시작하세요.")
        self.file_label.setObjectName("muted")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.file_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        browser, browser_layout = card()
        browser.setMinimumWidth(265)
        label = QLabel("데이터 행 탐색")
        label.setObjectName("section")
        browser_layout.addWidget(label)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("전체 행", -1)
        for i, name in enumerate(CLASS_NAMES):
            self.filter_combo.addItem(f"정답 {i} · {name}", i)
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)
        browser_layout.addWidget(self.filter_combo)
        self.dataset_summary = QLabel("진단 결과 대기 중")
        self.dataset_summary.setWordWrap(True)
        self.dataset_summary.setObjectName("muted")
        browser_layout.addWidget(self.dataset_summary)
        self.row_table = QTableWidget(0, 4)
        self.row_table.setHorizontalHeaderLabels(["행", "정답", "예측", "최대 확률"])
        self.row_table.setAlternatingRowColors(True)
        self.row_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.row_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.row_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.row_table.verticalHeader().hide()
        self.row_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.row_table.itemSelectionChanged.connect(self._table_selection)
        browser_layout.addWidget(self.row_table, 1)
        foot = QLabel("행 번호: 헤더를 제외한 CSV 데이터 행\n정답: fault_id  /  예측: 모델 출력")
        foot.setObjectName("muted")
        foot.setWordWrap(True)
        browser_layout.addWidget(foot)
        splitter.addWidget(browser)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        splitter.addWidget(right)
        splitter.setSizes([300, 1080])

        result_card, result_layout = card()
        result_card.setMinimumHeight(82)
        result_top = QHBoxLayout()
        prediction_column = QVBoxLayout()
        self.prediction_label = QLabel("진단 결과 대기")
        self.prediction_label.setObjectName("prediction")
        prediction_column.addWidget(self.prediction_label)
        self.actual_label = QLabel("정답 fault_id: —")
        self.actual_label.setObjectName("muted")
        prediction_column.addWidget(self.actual_label)
        result_top.addLayout(prediction_column, 1)
        self.match_label = QLabel("—")
        self.match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.match_label.setMinimumWidth(115)
        result_top.addWidget(self.match_label)
        result_layout.addLayout(result_top)
        right_layout.addWidget(result_card)

        wave_card, wave_layout = card()
        wave_card.setMinimumHeight(310)
        wave_heading = QHBoxLayout()
        wave_title = QLabel("입력 전류 파형")
        wave_title.setObjectName("section")
        wave_heading.addWidget(wave_title)
        wave_heading.addStretch()
        self.row_label = QLabel("CSV 행 —")
        self.row_label.setObjectName("muted")
        wave_heading.addWidget(self.row_label)
        wave_layout.addLayout(wave_heading)
        self.canvas = WaveformCanvas()
        self.canvas.setMinimumHeight(150)
        wave_layout.addWidget(self.canvas, 1)
        plot_toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_toolbar.setMaximumHeight(30)
        # 확대/저장 도구를 제목 행에 배치해 작은 창에서도 x축과 겹치지 않게 합니다.
        wave_heading.insertWidget(1, plot_toolbar)
        navigation = QHBoxLayout()
        self.prev_button = QPushButton("◀ 이전")
        self.next_button = QPushButton("다음 ▶")
        self.prev_button.clicked.connect(lambda: self.row_spin.setValue(self.row_spin.value() - 1))
        self.next_button.clicked.connect(lambda: self.row_spin.setValue(self.row_spin.value() + 1))
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 1)
        self.row_spin.valueChanged.connect(self._navigation_change)
        self.row_slider = QSlider(Qt.Orientation.Horizontal)
        self.row_slider.setRange(1, 1)
        self.row_slider.valueChanged.connect(self.row_spin.setValue)
        navigation.addWidget(self.prev_button)
        navigation.addWidget(self.row_slider, 1)
        navigation.addWidget(self.row_spin)
        self.filtered_count = QLabel("/ 0")
        navigation.addWidget(self.filtered_count)
        navigation.addWidget(self.next_button)
        wave_layout.addLayout(navigation)
        right_layout.addWidget(wave_card, 5)

        probabilities_card, probabilities_layout = card()
        probabilities_card.setMinimumHeight(316)
        probability_heading = QHBoxLayout()
        probability_title = QLabel("클래스별 Softmax")
        probability_title.setObjectName("section")
        probability_heading.addWidget(probability_title)
        probability_heading.addStretch()
        self.sum_label = QLabel("확률 합: —")
        self.sum_label.setObjectName("muted")
        probability_heading.addWidget(self.sum_label)
        probabilities_layout.addLayout(probability_heading)
        self.probability_table = QTableWidget(7, 5)
        self.probability_table.setHorizontalHeaderLabels(["ID", "진단 클래스", "Softmax (0–1)", "확률 (%)", "분포"])
        self.probability_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.probability_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.probability_table.verticalHeader().hide()
        self.probability_table.setShowGrid(False)
        self.probability_table.setMinimumHeight(241)
        self.probability_table.setMaximumHeight(260)
        header = self.probability_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.probability_bars = []
        for i, name in enumerate(CLASS_NAMES):
            self.probability_table.setRowHeight(i, 29)
            self.probability_table.setItem(i, 0, table_item(i, True))
            self.probability_table.setItem(i, 1, table_item(name))
            self.probability_table.setItem(i, 2, table_item("—", True))
            self.probability_table.setItem(i, 3, table_item("—", True))
            container = QWidget()
            bar_layout = QHBoxLayout(container)
            bar_layout.setContentsMargins(5, 8, 5, 8)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar_layout.addWidget(bar)
            self.probability_table.setCellWidget(i, 4, container)
            self.probability_bars.append(bar)
        probabilities_layout.addWidget(self.probability_table)
        note = QLabel("Softmax는 모델의 상대 출력 확률입니다. 정답 일치 여부와 별도로 해석하세요.")
        note.setObjectName("muted")
        probabilities_layout.addWidget(note)
        right_layout.addWidget(probabilities_card, 3)

        summary = str(getattr(self.engine, "model_summary", "CPU inference"))
        self.model_label = QLabel(summary)
        self.model_label.setObjectName("muted")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        self.statusBar().showMessage("준비 완료 · 모든 추론은 이 PC에서 수행됩니다.")

    def _set_busy(self, busy):
        has_data = self.dataset is not None and self.probabilities is not None
        self.open_button.setEnabled(not busy)
        self.sample_button.setEnabled(not busy)
        self.rerun_button.setEnabled(not busy and has_data)
        self.export_button.setEnabled(not busy and has_data)
        self.filter_combo.setEnabled(not busy and has_data and self.dataset.labels is not None)
        self.row_table.setEnabled(not busy and has_data)
        for widget in (self.row_spin, self.row_slider, self.prev_button, self.next_button):
            widget.setEnabled(not busy and has_data and len(self.visible_rows) > 0)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, 0)

    @property
    def is_busy(self):
        return self.worker is not None

    @property
    def busy(self):
        return self.is_busy

    def _clear_results(self):
        self.dataset = None
        self.probabilities = None
        self.visible_rows = np.array([], dtype=int)
        self.current_row = -1
        self.row_table.setRowCount(0)
        self.dataset_summary.setText("진단 결과 대기 중")
        self.prediction_label.setText("진단 결과 대기")
        self.actual_label.setText("정답 fault_id: —")
        self.match_label.setText("—")
        self.match_label.setStyleSheet("")
        self.row_label.setText("CSV 행 —")
        self.sum_label.setText("확률 합: —")
        self.filtered_count.setText("/ 0")
        self.canvas.clear_waveforms()
        for row, bar in enumerate(self.probability_bars):
            bar.setValue(0)
            for col in (2, 3):
                self.probability_table.item(row, col).setText("—")
            for col in range(4):
                self.probability_table.item(row, col).setBackground(QColor("white"))

    def open_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "전류 CSV 열기", "", "CSV 파일 (*.csv);;모든 파일 (*)")
        if filename:
            self.load_path(filename)

    def open_sample(self):
        self.load_path(resource_path("assets/sample.csv"))

    def rerun(self):
        if self.dataset is not None:
            self.load_path(self.dataset.path)

    def load_path(self, path):
        """새 파일을 열 때 기존 결과를 먼저 지워 이전 결과와 혼동하지 않습니다."""
        if self.worker is not None:
            return False
        self.last_error = ""
        self._clear_results()
        path = Path(path)
        self.file_label.setText(f"읽는 중: {path.name}")
        self.file_label.setToolTip(str(path))
        self._set_busy(True)
        self.statusBar().showMessage("CSV 읽기 및 전체 행 추론 중…")
        self.worker = InferenceWorker(self.engine, path, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.failed.connect(self._on_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()
        return True

    def _on_progress(self, done, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.statusBar().showMessage(f"진단 중: {done:,} / {total:,}행")

    def _on_result(self, dataset, probabilities):
        self.dataset = dataset
        self.probabilities = probabilities
        count = len(probabilities)
        self.file_label.setText(f"파일: {Path(dataset.path).name}  ·  {count:,}행  ·  전류 600개/행")
        predicted = probabilities.argmax(axis=1)
        if dataset.labels is not None:
            correct = int(np.sum(predicted == dataset.labels))
            self.dataset_summary.setText(f"전체 {count:,}행  ·  정답 일치 {correct:,}행\n이 CSV의 일치율: {correct / count:.2%}")
        else:
            self.dataset_summary.setText(f"전체 {count:,}행  ·  정답 레이블 없음")
        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentIndex(0)
        self.filter_combo.blockSignals(False)
        self.apply_filter()
        self.statusBar().showMessage(f"진단 완료 · {count:,}행 · 행을 선택하면 파형과 확률을 확인할 수 있습니다.")

    def _on_error(self, message):
        self.last_error = message
        self.file_label.setText("파일을 처리하지 못했습니다. 다른 CSV를 선택해 주세요.")
        self.statusBar().showMessage(f"오류: {message}")
        QMessageBox.warning(self, "진단 오류", message)

    def _worker_finished(self):
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def apply_filter(self):
        if self.dataset is None or self.probabilities is None:
            return
        selected_class = self.filter_combo.currentData()
        if selected_class == -1 or self.dataset.labels is None:
            self.visible_rows = np.arange(len(self.probabilities))
        else:
            self.visible_rows = np.flatnonzero(self.dataset.labels == selected_class)
        self._updating = True
        self.row_table.setRowCount(len(self.visible_rows))
        predictions = self.probabilities.argmax(axis=1)
        for pos, row in enumerate(self.visible_rows):
            actual = "—" if self.dataset.labels is None else int(self.dataset.labels[row])
            predicted = int(predictions[row])
            for col, value in enumerate((row + 1, actual, predicted, f"{self.probabilities[row, predicted]:.2%}")):
                item = table_item(value, True)
                if col == 2 and actual != "—" and predicted != actual:
                    item.setForeground(QColor("#b25c22"))
                    item.setToolTip("정답과 모델 예측이 다릅니다.")
                self.row_table.setItem(pos, col, item)
        total = len(self.visible_rows)
        self.row_spin.setRange(1, max(1, total))
        self.row_slider.setRange(1, max(1, total))
        self.row_spin.setValue(1)
        self.row_slider.setValue(1)
        self.filtered_count.setText(f"/ {total:,}")
        self._updating = False
        if total:
            self.set_row(int(self.visible_rows[0]))
        else:
            self.current_row = -1
            self.canvas.clear_waveforms()
            self.prediction_label.setText("선택한 클래스의 행이 없습니다")
            self.actual_label.setText("정답 fault_id: —")
            self.match_label.setText("—")
            self.match_label.setStyleSheet("")
            self.row_label.setText("CSV 행 —")
            self.sum_label.setText("확률 합: —")
            for row, bar in enumerate(self.probability_bars):
                bar.setValue(0)
                for col in (2, 3):
                    self.probability_table.item(row, col).setText("—")
        self._set_busy(self.worker is not None)

    def _navigation_change(self, position):
        if not self._updating and 1 <= position <= len(self.visible_rows):
            self.set_row(int(self.visible_rows[position - 1]))

    def _table_selection(self):
        position = self.row_table.currentRow()
        if not self._updating and 0 <= position < len(self.visible_rows):
            self.set_row(int(self.visible_rows[position]))

    def set_row(self, index):
        """0-based 원본 CSV 데이터 행을 표시합니다."""
        if self.dataset is None or self.probabilities is None:
            return
        if index < 0 or index >= len(self.probabilities):
            return
        positions = np.flatnonzero(self.visible_rows == index)
        if not len(positions):
            self.filter_combo.setCurrentIndex(0)
            positions = np.flatnonzero(self.visible_rows == index)
        if not len(positions):
            return
        position = int(positions[0])
        self.current_row = index
        self._updating = True
        self.row_spin.setValue(position + 1)
        self.row_slider.setValue(position + 1)
        self.row_table.selectRow(position)
        self._updating = False
        self.canvas.show_currents(self.dataset.currents[index])
        self.row_label.setText(f"CSV 데이터 행 {index + 1:,}  ·  샘플 0–199")
        probabilities = self.probabilities[index]
        predicted = int(probabilities.argmax())
        self.prediction_label.setText(f"{predicted} · {CLASS_NAMES[predicted]}  |  {probabilities[predicted]:.2%}")
        if self.dataset.labels is None:
            self.actual_label.setText("정답 fault_id: 없음")
            self.match_label.setText("정답 없음")
            self.match_label.setStyleSheet("color: #65768a; font-size: 15px; font-weight: 600;")
        else:
            actual = int(self.dataset.labels[index])
            self.actual_label.setText(f"정답 fault_id: {actual} · {CLASS_NAMES[actual]}")
            correct = predicted == actual
            self.match_label.setText("정답 일치" if correct else "정답 불일치")
            color = "#26745c" if correct else "#a25820"
            self.match_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 700;")
        self.sum_label.setText(f"확률 합: {float(probabilities.sum()):.6f}")
        for row, (value, bar) in enumerate(zip(probabilities, self.probability_bars)):
            # 표시 반올림과 별개로 원래 softmax 수치는 CSV 내보내기에 보존합니다.
            self.probability_table.item(row, 2).setText(f"{value:.8g}")
            self.probability_table.item(row, 3).setText(f"{value * 100:.4f}%")
            bar.setValue(int(round(float(value) * 10000)))
            color = "#245a8b" if row == predicted else "#a5bbcf"
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}")
            for col in range(4):
                item = self.probability_table.item(row, col)
                item.setBackground(QColor("#eaf2fa" if row == predicted else "white"))
                font = item.font()
                font.setBold(row == predicted)
                item.setFont(font)
        self.prev_button.setEnabled(position > 0 and self.worker is None)
        self.next_button.setEnabled(position + 1 < len(self.visible_rows) and self.worker is None)

    def export_results(self, checked=False, path=None):
        """각 행의 예측/정답/일치 여부와 softmax 7개를 UTF-8 CSV로 저장합니다."""
        if self.dataset is None or self.probabilities is None:
            return False
        if path is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "진단 결과 저장", Path(self.dataset.path).stem + "_diagnosis.csv", "CSV 파일 (*.csv)"
            )
            if not filename:
                return False
            path = Path(filename)
        else:
            path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".csv")
        try:
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["source_file", "data_row", "true_fault_id", "true_class", "predicted_fault_id", "predicted_class", "is_correct", "max_probability"] + [f"softmax_{i}" for i in range(7)])
                for row, values in enumerate(self.probabilities):
                    predicted = int(values.argmax())
                    actual = None if self.dataset.labels is None else int(self.dataset.labels[row])
                    writer.writerow([
                        Path(self.dataset.path).name, row + 1,
                        "" if actual is None else actual,
                        "" if actual is None else CLASS_NAMES[actual],
                        predicted, CLASS_NAMES[predicted],
                        "" if actual is None else int(actual == predicted),
                        format(float(values[predicted]), ".9g"),
                    ] + [format(float(value), ".9g") for value in values])
            self.statusBar().showMessage(f"결과 저장 완료: {path}")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            QMessageBox.warning(self, "저장 오류", str(exc))
            return False

    def closeEvent(self, event):
        if self.worker is not None:
            # 스레드를 강제 종료하지 않고 현재 배치 후 안전하게 마칩니다.
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)
