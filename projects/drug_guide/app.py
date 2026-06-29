"""
Drug Prescription Assistant — PyQt5
Scalable: load any compatible .pkl + .yaml model at runtime.
Run: python app.py
"""
import sys
import joblib
import yaml
import numpy as np
from pathlib import Path

from sklearn.tree import _tree

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar,
    QSizePolicy, QMessageBox, QFileDialog, QDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

HERE = Path(__file__).parent

# Colour palette assigned to classes in alphabetical order
_PALETTE = [
    ("#1565C0", "#E3F2FD"),
    ("#2E7D32", "#E8F5E9"),
    ("#E65100", "#FFF3E0"),
    ("#6A1B9A", "#F3E5F5"),
    ("#B71C1C", "#FFEBEE"),
    ("#00695C", "#E0F2F1"),
    ("#4527A0", "#EDE7F6"),
    ("#F57F17", "#FFFDE7"),
]

INPUT_QSS = """
    QDoubleSpinBox, QComboBox {
        border: 1.5px solid #ddd; border-radius: 6px;
        padding: 5px 10px; font-size: 13px;
        background: white; min-height: 34px; color: #222;
    }
    QDoubleSpinBox:focus, QComboBox:focus { border-color: #1a3a8f; }
    QComboBox::drop-down { border: none; width: 18px; }
"""


# ── Model helpers ──────────────────────────────────────────────────────────────
def _load_pkl(path: Path) -> dict:
    return joblib.load(path)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _class_colors(classes: list) -> dict:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(sorted(classes))}


def _decode(feature: str, threshold: float, direction: str, cat_decoders: dict) -> str:
    if feature not in cat_decoders:
        op = "<=" if direction == "left" else ">"
        return f"{feature} {op} {threshold:.3f}"
    labels = cat_decoders[feature]
    idx = int(threshold)
    matched = labels[: idx + 1] if direction == "left" else labels[idx + 1 :]
    return f"{feature} = {matched[0]}" if len(matched) == 1 else f"{feature} in [{', '.join(matched)}]"


def _decision_path(clf, feature_names: list, cat_decoders: dict, x: np.ndarray):
    """Returns (steps, confidence). Works only for Decision Trees."""
    if not hasattr(clf, "tree_"):
        return [], None
    t, node = clf.tree_, 0
    steps = []
    while t.children_left[node] != _tree.TREE_LEAF:
        fi  = t.feature[node]
        thr = t.threshold[node]
        val = x[0][fi]
        left = val <= thr
        steps.append(_decode(feature_names[fi], thr, "left" if left else "right", cat_decoders))
        node = t.children_left[node] if left else t.children_right[node]
    vals = t.value[node][0]
    return steps, float(vals.max() / vals.sum())


# ── Model info dialog ─────────────────────────────────────────────────────────
class InfoDialog(QDialog):
    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Information")
        self.setMinimumWidth(480)
        info = meta.get("model", {})

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(20, 20, 20, 20)

        def row(key, val):
            f = QFrame()
            fl = QHBoxLayout(f)
            fl.setContentsMargins(0, 0, 0, 0)
            k = QLabel(key)
            k.setStyleSheet("font-weight: bold; color: #555; min-width: 110px;")
            v = QLabel(str(val))
            v.setWordWrap(True)
            v.setStyleSheet("color: #222;")
            fl.addWidget(k)
            fl.addWidget(v, 1)
            lay.addWidget(f)

        row("Name",      info.get("name", "—"))
        row("Version",   info.get("version", "—"))
        row("Trained on",info.get("trained_on", "—"))
        row("Algorithm", info.get("algorithm", "—"))
        row("Dataset",   info.get("dataset", "—"))
        acc = info.get("accuracy")
        row("Accuracy",  f"{acc:.2%}" if acc else "—")

        if info.get("notes"):
            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #ddd;"); lay.addWidget(sep)
            notes_lbl = QLabel(info["notes"])
            notes_lbl.setWordWrap(True)
            notes_lbl.setStyleSheet("color: #333; font-size: 12px;")
            lay.addWidget(notes_lbl)

        for w in info.get("warnings", []):
            wf = QFrame()
            wf.setStyleSheet(
                "QFrame { background: #FFF8E1; border-left: 3px solid #F9A825; } "
                "QLabel { background: transparent; color: #555; font-size: 12px; }"
            )
            wl = QVBoxLayout(wf); wl.setContentsMargins(10, 6, 10, 6)
            wlbl = QLabel(f"⚠  {w}"); wlbl.setWordWrap(True)
            wl.addWidget(wlbl); lay.addWidget(wf)

        drugs = info.get("drugs", {})
        if drugs:
            sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
            sep2.setStyleSheet("color: #ddd;"); lay.addWidget(sep2)
            lay.addWidget(QLabel("<b>Class / Drug Reference</b>"))
            for name, desc in sorted(drugs.items()):
                dl = QLabel(f"<b>{name.upper()}</b> — {desc}")
                dl.setWordWrap(True)
                dl.setStyleSheet("font-size: 12px; color: #333;")
                lay.addWidget(dl)

        close = QPushButton("Close")
        close.setStyleSheet(
            "QPushButton { background:#1a3a8f; color:white; border:none; border-radius:6px; "
            "padding:8px 20px; font-size:13px; }"
            "QPushButton:hover { background:#1565C0; }"
        )
        close.clicked.connect(self.accept)
        lay.addStretch()
        lay.addWidget(close, alignment=Qt.AlignRight)


# ── Result card ────────────────────────────────────────────────────────────────
class ResultCard(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("ResultCard")
        self.setVisible(False)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(22, 18, 22, 18)
        self._lay.setSpacing(6)

    def _clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_result(self, label: str, confidence: float | None, steps: list, desc: str, color: str, light: str):
        self._clear()
        self.setStyleSheet(
            f"QFrame#ResultCard {{ background:{light}; border-radius:12px; border-left:6px solid {color}; }}"
            "QLabel { background:transparent; }"
        )

        tag = QLabel("RESULT")
        tag.setStyleSheet("color:rgba(0,0,0,0.42); font-size:10px; font-weight:bold; letter-spacing:2px;")
        self._lay.addWidget(tag)

        name_lbl = QLabel(label.upper())
        name_lbl.setStyleSheet(f"color:{color}; font-size:34px; font-weight:900; letter-spacing:-1px;")
        self._lay.addWidget(name_lbl)

        if desc:
            d = QLabel(desc); d.setWordWrap(True)
            d.setStyleSheet("color:#444; font-size:13px;")
            self._lay.addWidget(d)

        self._lay.addSpacing(10)

        if confidence is not None:
            conf_lbl = QLabel(f"Model confidence: {confidence:.1%}")
            conf_lbl.setStyleSheet("font-size:12px; font-weight:bold; color:#333;")
            self._lay.addWidget(conf_lbl)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(confidence * 100))
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(
                f"QProgressBar {{ background:rgba(0,0,0,0.10); border-radius:4px; border:none; }}"
                f"QProgressBar::chunk {{ background:{color}; border-radius:4px; }}"
            )
            self._lay.addWidget(bar)

        if steps:
            self._lay.addSpacing(14)
            why = QLabel("WHY THIS RESULT?")
            why.setStyleSheet("color:rgba(0,0,0,0.42); font-size:10px; font-weight:bold; letter-spacing:2px;")
            self._lay.addWidget(why)
            sub = QLabel("Decision path through the model:")
            sub.setStyleSheet("color:#555; font-size:12px;")
            self._lay.addWidget(sub)
            self._lay.addSpacing(4)

            for i, step in enumerate(steps, 1):
                row = QFrame()
                row.setStyleSheet(
                    "QFrame { background:rgba(0,0,0,0.04); border-radius:6px; }"
                    "QLabel { background:transparent; }"
                )
                rl = QHBoxLayout(row); rl.setContentsMargins(10, 7, 10, 7); rl.setSpacing(10)
                num = QLabel(str(i))
                num.setFixedSize(24, 24); num.setAlignment(Qt.AlignCenter)
                num.setStyleSheet(f"background:{color}; color:white; border-radius:12px; font-size:11px; font-weight:bold;")
                rl.addWidget(num)
                txt = QLabel(step); txt.setStyleSheet("color:#222; font-size:13px;")
                rl.addWidget(txt); rl.addStretch()
                self._lay.addWidget(row)

        self._lay.addSpacing(12)
        disc = QFrame()
        disc.setStyleSheet(
            "QFrame { background:#FFF8E1; border-left:4px solid #F9A825; border-radius:0px 8px 8px 0px; }"
            "QLabel { background:transparent; color:#666; font-size:12px; }"
        )
        dl = QVBoxLayout(disc); dl.setContentsMargins(12, 8, 12, 8)
        d = QLabel("⚠  Clinical Disclaimer: This is a machine learning recommendation. Always apply clinical judgment before prescribing.")
        d.setWordWrap(True); dl.addWidget(d)
        self._lay.addWidget(disc)
        self.setVisible(True)


# ── Main window ────────────────────────────────────────────────────────────────
class DrugApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.artifacts: dict | None = None
        self.meta: dict = {}
        self._inputs: dict = {}       # feature_name → widget
        self._color_map: dict = {}    # class_name → (color, light)

        self.setWindowTitle("Drug Prescription Assistant")
        self.setMinimumSize(780, 560)
        self.resize(900, 660)

        root = QWidget(); root.setStyleSheet("background:#F0F2F8;")
        self.setCentralWidget(root)
        self._root_lay = QVBoxLayout(root)
        self._root_lay.setContentsMargins(0, 0, 0, 0)
        self._root_lay.setSpacing(0)

        self._build_topbar()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background:#F0F2F8; border:none; }")
        self._root_lay.addWidget(scroll)

        self._scroll_content = QWidget(); self._scroll_content.setStyleSheet("background:#F0F2F8;")
        self._scroll_lay = QVBoxLayout(self._scroll_content)
        self._scroll_lay.setContentsMargins(28, 24, 28, 28)
        self._scroll_lay.setSpacing(16)
        scroll.setWidget(self._scroll_content)

        self._form_card = self._make_card()
        self._form_outer = QVBoxLayout(self._form_card)
        self._form_outer.setContentsMargins(24, 20, 24, 20)
        self._form_outer.setSpacing(14)
        self._scroll_lay.addWidget(self._form_card)

        self._result = ResultCard()
        self._scroll_lay.addWidget(self._result)
        self._scroll_lay.addStretch()

        self._build_empty_state()

        # Auto-load if default model exists
        default = HERE / "drug_model.pkl"
        if default.exists():
            self._load_from(default)

    # ── Top bar ───────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = QFrame()
        bar.setFixedHeight(58)
        bar.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0d1b4b,stop:1 #1a3a8f); }"
            "QLabel { background:transparent; color:white; }"
            "QPushButton { border:none; }"
        )
        bl = QHBoxLayout(bar); bl.setContentsMargins(20, 0, 16, 0); bl.setSpacing(12)

        title = QLabel("⚕  Drug Prescription Assistant")
        title.setStyleSheet("font-size:16px; font-weight:800; color:white;")
        bl.addWidget(title)
        bl.addStretch()

        self._model_chip = QLabel()
        self._model_chip.setVisible(False)
        self._model_chip.setStyleSheet(
            "background:rgba(255,255,255,0.14); color:white; border-radius:12px; "
            "padding:3px 12px; font-size:11px; font-weight:600;"
        )
        bl.addWidget(self._model_chip)

        info_btn = QPushButton("ℹ")
        info_btn.setFixedSize(32, 32)
        info_btn.setObjectName("info-btn")
        info_btn.setVisible(False)
        info_btn.setToolTip("Model information")
        info_btn.setStyleSheet(
            "QPushButton#info-btn { background:rgba(255,255,255,0.15); color:white; "
            "border-radius:16px; font-size:14px; }"
            "QPushButton#info-btn:hover { background:rgba(255,255,255,0.25); }"
        )
        info_btn.clicked.connect(self._show_info)
        self._info_btn = info_btn
        bl.addWidget(info_btn)

        load_btn = QPushButton("📂  Load Model")
        load_btn.setFixedHeight(34)
        load_btn.setStyleSheet(
            "QPushButton { background:rgba(255,255,255,0.18); color:white; border:1px solid rgba(255,255,255,0.35); "
            "border-radius:8px; font-size:13px; font-weight:600; padding:0 14px; }"
            "QPushButton:hover { background:rgba(255,255,255,0.28); }"
        )
        load_btn.clicked.connect(self._browse)
        bl.addWidget(load_btn)

        self._root_lay.addWidget(bar)

    # ── Empty state ───────────────────────────────────────────────────────────
    def _build_empty_state(self):
        self._empty = QLabel("No model loaded.\nClick  📂 Load Model  to begin.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color:#aaa; font-size:15px; padding:40px;")
        self._form_outer.addWidget(self._empty)

    # ── Load model ────────────────────────────────────────────────────────────
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model file", str(HERE), "Pickle files (*.pkl);;All files (*)"
        )
        if path:
            self._load_from(Path(path))

    def _load_from(self, pkl_path: Path):
        try:
            arts = _load_pkl(pkl_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load model:\n{e}")
            return

        required = {"clf", "cat_decoders", "feature_names"}
        if not required.issubset(arts):
            QMessageBox.warning(self, "Invalid Model",
                f"The file is missing required keys: {required - arts.keys()}\n\n"
                "Export the model using the notebook's Section 7 cell.")
            return

        self.artifacts = arts
        self.meta = _load_yaml(pkl_path.with_suffix(".yaml").with_name(pkl_path.stem.replace("model", "meta") + ".yaml"))
        # also try <stem>_meta.yaml and model_meta.yaml
        if not self.meta:
            for candidate in [
                pkl_path.parent / "model_meta.yaml",
                pkl_path.parent / (pkl_path.stem + "_meta.yaml"),
            ]:
                self.meta = _load_yaml(candidate)
                if self.meta:
                    break

        classes = arts.get("classes", [str(c) for c in arts["clf"].classes_])
        self._color_map = _class_colors(classes)

        acc = self.meta.get("model", {}).get("accuracy")
        acc_str = f"  ·  {acc:.2%}" if acc else ""
        self._model_chip.setText(f"  {pkl_path.name}{acc_str}  ")
        self._model_chip.setVisible(True)
        self._info_btn.setVisible(True)

        self._rebuild_form()
        self._result.setVisible(False)

    # ── Dynamic form ──────────────────────────────────────────────────────────
    def _rebuild_form(self):
        # Clear layout
        while self._form_outer.count():
            item = self._form_outer.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
        self._inputs.clear()

        sec = QLabel("PATIENT DATA")
        sec.setStyleSheet("color:#999; font-size:10px; font-weight:bold; letter-spacing:2px;")
        self._form_outer.addWidget(sec)

        cat  = self.artifacts["cat_decoders"]
        feats = self.artifacts["feature_names"]
        COLS = 3

        grid = QGridLayout(); grid.setSpacing(14)
        for col in range(COLS):
            grid.setColumnStretch(col, 1)

        for i, feat in enumerate(feats):
            r, c = (i // COLS) * 2, i % COLS

            lbl = QLabel(feat.replace("_", " "))
            lbl.setStyleSheet("font-weight:600; font-size:13px; color:#333;")
            grid.addWidget(lbl, r, c)

            if feat in cat:
                w = QComboBox()
                w.addItems(cat[feat])
                w.setStyleSheet(INPUT_QSS)
                w.setMinimumHeight(36)
            else:
                w = QDoubleSpinBox()
                w.setRange(0.0, 100_000.0)
                w.setSingleStep(0.1)
                w.setDecimals(3)
                w.setStyleSheet(INPUT_QSS)
                w.setMinimumHeight(36)

            self._inputs[feat] = w
            grid.addWidget(w, r + 1, c)

        self._form_outer.addLayout(grid)
        self._form_outer.addSpacing(4)

        btn = QPushButton("Analyze Patient")
        btn.setFixedHeight(46)
        btn.setStyleSheet(
            "QPushButton { background:#1a3a8f; color:white; border:none; border-radius:8px; "
            "font-size:14px; font-weight:bold; }"
            "QPushButton:hover  { background:#1565C0; }"
            "QPushButton:pressed{ background:#0d47a1; }"
        )
        btn.clicked.connect(self._analyze)
        self._form_outer.addWidget(btn)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _analyze(self):
        if not self.artifacts:
            return

        clf          = self.artifacts["clf"]
        cat          = self.artifacts["cat_decoders"]
        feature_names = self.artifacts["feature_names"]

        row = []
        for feat in feature_names:
            w = self._inputs[feat]
            if feat in cat:
                row.append(cat[feat].index(w.currentText()))
            else:
                row.append(w.value())

        x     = np.array([row])
        label = clf.predict(x)[0]
        steps, confidence = _decision_path(clf, feature_names, cat, x)

        # If no decision path, try predict_proba for confidence
        if confidence is None and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x)[0]
            confidence = float(proba.max())

        color, light = self._color_map.get(str(label), ("#333", "#F5F5F5"))
        desc = self.meta.get("model", {}).get("drugs", {}).get(str(label), "")

        self._result.show_result(str(label), confidence, steps, desc, color, light)

    # ── Info dialog ───────────────────────────────────────────────────────────
    def _show_info(self):
        dlg = InfoDialog(self.meta, self)
        dlg.exec_()

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _make_card() -> QFrame:
        f = QFrame()
        f.setStyleSheet(
            "QFrame { background:white; border-radius:12px; border:1px solid #e0e0e0; }"
            "QLabel { background:transparent; }"
        )
        return f


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = DrugApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
