#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PySide6 is required for the Celeste setup wizard. Install dependencies first:\n"
        "  pip install -r requirements.txt"
    ) from exc

from validate_environment import format_checks, has_errors, validate_config_file


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _template_path() -> str:
    return os.path.join(_project_root(), "config.example.yaml")


def _default_paths() -> dict[str, str]:
    home = os.path.expanduser("~")
    celeste_home = os.path.join(home, "Celeste")
    return {
        "model_path": os.path.join(celeste_home, "models", "your-model.gguf"),
        "embedding_model": os.path.join(celeste_home, "embeddings", "e5-small-v2"),
        "llama_server_executable": os.path.join(
            _project_root(),
            "vendor",
            "llama.cpp",
            "build",
            "bin",
            "llama-server.exe" if os.name == "nt" else "llama-server",
        ),
        "data_dir": os.path.join(celeste_home, "data"),
        "persist_dir": os.path.join(celeste_home, "chroma"),
        "file_rag_dir": os.path.join(celeste_home, "library"),
        "tts_piper_model": os.path.join(celeste_home, "voices", "voice.onnx"),
        "tts_piper_config": os.path.join(celeste_home, "voices", "voice.onnx.json"),
        "tts_piper_executable": "piper",
    }


def _load_template(template_path: str | None = None) -> dict:
    with open(template_path or _template_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SetupWizardDialog(QDialog):
    def __init__(self, config_path: str, template_path: str | None = None, parent=None):
        super().__init__(parent)
        self.config_path = os.path.abspath(config_path)
        self.template = _load_template(template_path)
        self.defaults = _default_paths()
        self.inputs: dict[str, QLineEdit] = {}
        self.setWindowTitle("Celeste First-Run Setup")
        self.resize(900, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Celeste Setup Wizard")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        intro = QLabel(
            "Choose local model, embedding, document, and data paths. "
            "You can keep the suggested default folders or browse to your own existing paths."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        use_defaults = QPushButton("Use Default Celeste Folders")
        use_defaults.clicked.connect(self._apply_default_paths)
        layout.addWidget(use_defaults, alignment=Qt.AlignLeft)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form_layout = QVBoxLayout(content)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        self.inputs["model_path"] = self._add_path_row(form, "GGUF Model", self.defaults["model_path"], "file")
        self.inputs["embedding_model"] = self._add_path_row(
            form,
            "Embedding Model",
            self.defaults["embedding_model"],
            "directory",
        )
        self.inputs["llama_server_executable"] = self._add_path_row(
            form,
            "llama-server",
            self.defaults["llama_server_executable"],
            "file",
        )
        self.inputs["data_dir"] = self._add_path_row(form, "Data Directory", self.defaults["data_dir"], "directory")
        self.inputs["persist_dir"] = self._add_path_row(
            form,
            "Vector DB Directory",
            self.defaults["persist_dir"],
            "directory",
        )
        self.inputs["file_rag_dir"] = self._add_path_row(
            form,
            "Document Library",
            self.defaults["file_rag_dir"],
            "directory",
        )

        self.tts_enabled = QCheckBox("Enable Piper TTS")
        self.tts_enabled.setChecked(False)
        form.addRow("Speech", self.tts_enabled)

        self.inputs["tts_piper_executable"] = self._add_path_row(
            form,
            "Piper Executable",
            self.defaults["tts_piper_executable"],
            "file",
        )
        self.inputs["tts_piper_model"] = self._add_path_row(
            form,
            "Piper Voice Model",
            self.defaults["tts_piper_model"],
            "file",
        )
        self.inputs["tts_piper_config"] = self._add_path_row(
            form,
            "Piper Voice Config",
            self.defaults["tts_piper_config"],
            "file",
        )

        form_layout.addLayout(form)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.validation_view = QTextBrowser()
        self.validation_view.setFixedHeight(180)
        self.validation_view.setPlaceholderText("Validation output will appear here.")
        layout.addWidget(self.validation_view)

        button_row = QHBoxLayout()
        self.validate_button = QPushButton("Validate")
        self.save_button = QPushButton("Save Config")
        self.cancel_button = QPushButton("Cancel")
        self.validate_button.clicked.connect(self._validate_current_settings)
        self.save_button.clicked.connect(self._save_config)
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

    def _add_path_row(self, form: QFormLayout, label: str, value: str, mode: str) -> QLineEdit:
        edit = QLineEdit(value)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda _, target=edit, browse_mode=mode: self._browse_path(target, browse_mode))
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _browse_path(self, target: QLineEdit, mode: str) -> None:
        start = target.text().strip() or os.path.expanduser("~")
        if mode == "file":
            path, _ = QFileDialog.getOpenFileName(self, "Choose File", start)
        else:
            path = QFileDialog.getExistingDirectory(self, "Choose Directory", start)
        if path:
            target.setText(path)

    def _apply_default_paths(self) -> None:
        for key, value in self.defaults.items():
            if key in self.inputs:
                self.inputs[key].setText(value)

    def _collect_config(self) -> dict:
        cfg = copy.deepcopy(self.template)
        cfg["model_path"] = self.inputs["model_path"].text().strip()
        cfg["embedding_model"] = self.inputs["embedding_model"].text().strip()
        cfg["llama_server_executable"] = self.inputs["llama_server_executable"].text().strip()
        cfg["data_dir"] = self.inputs["data_dir"].text().strip()
        cfg["persist_dir"] = self.inputs["persist_dir"].text().strip()
        file_rag_dir = self.inputs["file_rag_dir"].text().strip()
        cfg["file_rag_dirs"] = [file_rag_dir] if file_rag_dir else []
        cfg["tts_enabled"] = self.tts_enabled.isChecked()
        cfg["tts_backend"] = "piper"
        cfg["tts_piper_executable"] = self.inputs["tts_piper_executable"].text().strip()
        cfg["tts_piper_model"] = self.inputs["tts_piper_model"].text().strip()
        cfg["tts_piper_config"] = self.inputs["tts_piper_config"].text().strip()
        cfg["tts_output_dir"] = None
        return cfg

    def _write_config(self) -> None:
        cfg = self._collect_config()
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        for key in ("data_dir", "persist_dir"):
            directory = os.path.expandvars(os.path.expanduser(str(cfg.get(key, "") or "")))
            if directory:
                os.makedirs(directory, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

    def _validate_current_settings(self) -> bool:
        self._write_config()
        try:
            checks = validate_config_file(self.config_path)
        except Exception as exc:
            self.validation_view.setPlainText(f"[ERROR] config.yaml: {exc}")
            return False
        rendered = format_checks(checks)
        self.validation_view.setPlainText(rendered)
        return not has_errors(checks)

    def _save_config(self) -> None:
        valid = self._validate_current_settings()
        if valid:
            QMessageBox.information(self, "Celeste Setup", f"Configuration saved to:\n{self.config_path}")
            self.accept()
            return
        reply = QMessageBox.question(
            self,
            "Save With Validation Errors?",
            "Validation reported one or more errors. Save config.yaml anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.accept()


def ensure_config_with_wizard(config_path: str, *, app: QApplication | None = None, parent=None) -> bool:
    normalized = os.path.abspath(config_path)
    if os.path.isfile(normalized):
        return True
    dialog = SetupWizardDialog(normalized, parent=parent)
    return dialog.exec() == QDialog.Accepted


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a Celeste config.yaml with a first-run setup wizard.")
    parser.add_argument("--config", default="config.yaml", help="Path where config.yaml should be written.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Open the wizard even if the config file already exists.",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if os.path.exists(config_path) and not args.force:
        print(f"Config already exists: {config_path}")
        return 0

    app = QApplication.instance() or QApplication(sys.argv)
    dialog = SetupWizardDialog(config_path)
    return 0 if dialog.exec() == QDialog.Accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
