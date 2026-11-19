"""SEED Fault Diagnosis 실행 진입점 및 배포본 검증 명령."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_API', 'pyqt6')


def main():
    parser = argparse.ArgumentParser(description='SEED Lab three-phase fault diagnosis')
    parser.add_argument('--csv', type=Path, help='Open this CSV instead of the bundled example')
    parser.add_argument('--self-test', type=Path, metavar='REPORT_JSON', help='Verify bundled assets and inference, then exit')
    parser.add_argument('--capture', type=Path, metavar='IMAGE_PNG', help='Render the GUI and save an automated QA capture, then exit')
    args = parser.parse_args()
    from diagnosis_core import DiagnosisEngine, load_csv, resource_path
    import numpy as np
    engine = DiagnosisEngine(resource_path('assets/fnn_fault_model.pt'))
    csv_path = args.csv or resource_path('assets/sample.csv')
    if args.self_test:
        dataset = load_csv(csv_path)
        probabilities = engine.predict(dataset.currents)
        assert probabilities.shape == (len(dataset.currents), 7)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        # Row inference must not depend on which other records were imported.
        single = engine.predict(dataset.currents[:1])
        assert np.allclose(single, probabilities[:1], atol=1e-6)
        pred = probabilities.argmax(1)
        report = {
            'status': 'passed', 'frozen_executable': bool(getattr(sys, 'frozen', False)),
            'model_summary': engine.model_summary, 'rows': len(pred),
            'input_shape': list(dataset.currents.shape), 'probability_shape': list(probabilities.shape),
            'probability_sum_max_error': float(np.abs(probabilities.sum(1)-1).max()),
            'single_vs_batch_max_error': float(np.abs(single-probabilities[:1]).max()),
            'predictions': pred.tolist(), 'probabilities': probabilities.tolist(),
        }
        if dataset.labels is not None:
            report['agreement_count'] = int((pred == dataset.labels).sum())
            report['agreement_percent'] = float((pred == dataset.labels).mean()*100)
            report['class_counts'] = np.bincount(dataset.labels, minlength=7).tolist()
            report['note'] = 'Bundled sample agreement only; this is not independent generalization accuracy.'
        args.self_test.parent.mkdir(parents=True, exist_ok=True)
        args.self_test.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from gui import MainWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName('SEED Fault Diagnosis')
    app.setOrganizationName('SEED Lab')
    app.setFont(QFont('Malgun Gothic', 10))

    def exception_hook(exc_type, value, tb):
        import traceback
        message = ''.join(traceback.format_exception(exc_type, value, tb))
        log = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'SEED_FaultDiagnosis' / 'error.log'
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(message, encoding='utf-8')
        except OSError:
            pass
        QMessageBox.critical(None, '오류', f'{value}\n\n오류 기록: {log}')
    sys.excepthook = exception_hook
    window = MainWindow(engine, auto_load_sample=False)
    window.show()
    QTimer.singleShot(0, lambda: window.load_path(csv_path))
    if args.capture:
        import time
        started = time.monotonic()
        def save_capture():
            if time.monotonic() - started > 60:
                app.exit(2)
                return
            if (getattr(window, 'probabilities', None) is None or getattr(window, 'busy', False)
                    or window.dataset.path.resolve() != csv_path.resolve()):
                QTimer.singleShot(100, save_capture)
                return
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            window.set_row(min(30, len(window.dataset.currents)-1))
            def grab():
                ok = window.grab().save(str(args.capture))
                app.exit(0 if ok else 3)
            QTimer.singleShot(400, grab)
        QTimer.singleShot(600, save_capture)
    return app.exec()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        import traceback
        error_text = traceback.format_exc()
        if '--self-test' in sys.argv or '--capture' in sys.argv:
            # windowed EXE has no stdout/stderr; keep QA failures observable.
            option = '--self-test' if '--self-test' in sys.argv else '--capture'
            out = Path(sys.argv[sys.argv.index(option)+1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix('.error.txt').write_text(error_text, encoding='utf-8')
        else:
            from PyQt6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv[:1])
            QMessageBox.critical(None, '시작 오류', f'프로그램을 시작할 수 없습니다.\n\n{error}')
        raise SystemExit(1)
