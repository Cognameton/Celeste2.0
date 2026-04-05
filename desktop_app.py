#!/usr/bin/env python3
from __future__ import annotations

import html
import os
import re
import sys

try:
    from PySide6.QtCore import QObject, QMetaObject, QThread, Qt, Signal, Slot
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QProgressBar,
        QProgressDialog,
        QSpinBox,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "PySide6 is not installed. Install it in the Celeste venv with:\n"
        "  pip install PySide6"
    ) from exc

from app_service import CelesteService
from config_types import AgentConfig
from app_paths import default_config_path, resource_path


class ServiceWorker(QObject):
    initialized = Signal(object)
    models_ready = Signal(list)
    reloaded = Signal(object)
    rag_updated = Signal(object, str)
    memory_updated = Signal(object, str)
    reply_ready = Signal(str, str, str)
    failed = Signal(str)
    status = Signal(str)
    progress = Signal(int, str)

    def __init__(self, config_path: str):
        super().__init__()
        self.service = CelesteService(config_path)

    @Slot()
    def initialize(self) -> None:
        try:
            self.status.emit("Starting Celeste backend...")
            cfg = self.service.start(status_cb=self.status.emit)
            self.initialized.emit(cfg)
            self.status.emit("Celeste ready.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def load_models(self) -> None:
        try:
            self.models_ready.emit(self.service.available_models())
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(str)
    def send_message(self, message: str) -> None:
        try:
            self.status.emit("Generating reply...")
            answer, critique, improvements = self.service.chat(message)
            self.reply_ready.emit(answer, critique or "", improvements or "")
            if answer.strip():
                self.status.emit("Speaking reply...")
                self.service.speak(answer)
            self.status.emit("Ready.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(object, bool)
    def reload_config(self, overrides: object, persist: bool) -> None:
        try:
            data = dict(overrides) if isinstance(overrides, dict) else {}
            self.status.emit("Reloading Celeste with updated settings...")
            cfg = self.service.reload(data, persist=persist, status_cb=self.status.emit)
            self.reloaded.emit(cfg)
            self.models_ready.emit(self.service.available_models())
            self.status.emit("Reload complete.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(str)
    def add_rag_directory(self, directory: str) -> None:
        try:
            active_path = os.path.abspath(os.path.expanduser(os.path.expandvars(directory)))
            self.status.emit(f"Indexing files under {active_path}...")
            cfg, stats = self.service.add_rag_directory(directory)
            message = (
                f"Indexed {stats.get('files_indexed', 0)} files into "
                f"{stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File index updated.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def reindex_rag(self) -> None:
        try:
            self.status.emit("Reindexing file knowledge base...")
            cfg, stats = self.service.reindex_rag()
            message = (
                f"Reindexed {stats.get('files_indexed', 0)} files into "
                f"{stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File index updated.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def build_deep_index(self) -> None:
        try:
            def on_progress(message: str, percent: int = 0) -> None:
                self.status.emit(message)
                self.progress.emit(int(percent), message)

            on_progress("Building deep library index...", 0)
            cfg, stats = self.service.build_deep_rag_index(progress_cb=on_progress)
            message = (
                f"Deep index ready: {stats.get('files_indexed', 0)} files, "
                f"{stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.progress.emit(100, "Deep index ready.")
            self.status.emit("Deep index ready.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(str)
    def remove_rag_directory(self, directory: str) -> None:
        try:
            self.status.emit(f"Removing {directory} from Celeste file access...")
            cfg, stats = self.service.remove_rag_directory(directory)
            message = (
                f"Removed directory. Remaining index contains "
                f"{stats.get('files_indexed', 0)} files across {stats.get('chunks_indexed', 0)} chunks."
            )
            self.rag_updated.emit(cfg, message)
            self.status.emit("File access updated.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot()
    def shutdown(self) -> None:
        self.service.shutdown()

    @Slot(object)
    def purge_engram_memory(self, seconds: object) -> None:
        try:
            purge_seconds = None if seconds is None else int(seconds)
            if purge_seconds is None:
                self.status.emit("Purging all Engram memory...")
            else:
                self.status.emit("Purging recent Engram memory...")
            cfg, stats = self.service.purge_engram_memory(seconds=purge_seconds)
            deleted = int(stats.get("purged", stats.get("entries_deleted", 0)))
            remaining = int(stats.get("entries", stats.get("entries_remaining", 0)))
            message = f"Purged {deleted} Engram entries. Remaining Engram entries: {remaining}."
            self.memory_updated.emit(cfg, message)
            self.status.emit("Engram memory updated.")
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(bool)
    def set_engram_auto_prune(self, enabled: bool) -> None:
        try:
            self.status.emit(
                "Enabling Engram auto-purge..." if enabled else "Disabling Engram auto-purge..."
            )
            cfg, stats = self.service.set_engram_auto_prune(bool(enabled))
            retention_days = int(stats.get("retention_days", 30))
            keep_min_uses = int(stats.get("keep_min_uses", 3))
            if enabled:
                message = (
                    f"Engram auto-purge enabled: entries older than {retention_days} days "
                    f"are pruned unless used at least {keep_min_uses} times."
                )
            else:
                message = "Engram auto-purge disabled."
            self.memory_updated.emit(cfg, message)
            self.status.emit("Engram memory settings updated.")
        except Exception as exc:
            self.failed.emit(str(exc))


class CelesteWindow(QMainWindow):
    initialize_requested = Signal()
    load_models_requested = Signal()
    send_requested = Signal(str)
    reload_requested = Signal(object, bool)
    add_rag_directory_requested = Signal(str)
    remove_rag_directory_requested = Signal(str)
    reindex_rag_requested = Signal()
    build_deep_index_requested = Signal()
    purge_engram_requested = Signal(object)
    set_engram_auto_prune_requested = Signal(bool)
    shutdown_requested = Signal()

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.cfg: AgentConfig | None = None
        self.busy = False
        self._busy_reason: str | None = None
        self._build_ui()
        self._build_worker()
        self.load_models_requested.emit()
        self.initialize_requested.emit()

    def _build_ui(self) -> None:
        self.setWindowTitle("Celeste")
        icon_path = resource_path("assets", "celeste_icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1280, 820)

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        settings_card = QFrame()
        settings_card.setObjectName("settingsCard")
        settings_card.setMinimumWidth(120)
        settings_card.setMaximumWidth(420)
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(14, 14, 14, 14)
        settings_layout.setSpacing(10)

        title = QLabel("Control Deck")
        title.setObjectName("panelTitle")
        settings_layout.addWidget(title)

        subtitle = QLabel("Switch models, toggle speech, and reload the backend without leaving the app.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.NoInsert)
        self.model_combo.setMinimumHeight(32)
        form.addRow("LLM Model", self.model_combo)

        self.tts_toggle = QCheckBox("Enable Piper speech")
        form.addRow("Speech", self.tts_toggle)

        self.memory_toggle = QCheckBox("Use vector memory")
        form.addRow("Memory", self.memory_toggle)

        self.reflection_toggle = QCheckBox("Enable reflection pass")
        form.addRow("Reflection", self.reflection_toggle)

        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(64, 8192)
        self.tokens_spin.setSingleStep(64)
        self.tokens_spin.setMinimumHeight(32)
        form.addRow("Max Tokens", self.tokens_spin)

        settings_layout.addLayout(form)

        engram_title = QLabel("Engram Memory")
        engram_title.setObjectName("panelTitle")
        settings_layout.addWidget(engram_title)

        engram_subtitle = QLabel(
            "Purge exact-recall Engram traces manually, or keep auto-purge enabled for stale entries."
        )
        engram_subtitle.setWordWrap(True)
        engram_subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(engram_subtitle)

        self.engram_auto_toggle = QCheckBox(
            "Auto-purge Engram entries older than 30 days unless frequently used"
        )
        settings_layout.addWidget(self.engram_auto_toggle)

        engram_row = QHBoxLayout()
        engram_row.setSpacing(10)
        self.engram_purge_combo = QComboBox()
        self.engram_purge_combo.addItem("Purge all", userData=None)
        self.engram_purge_combo.addItem("Purge last 24 hours", userData=24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 7 days", userData=7 * 24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 30 days", userData=30 * 24 * 60 * 60)
        self.engram_purge_combo.addItem("Purge last 90 days", userData=90 * 24 * 60 * 60)
        self.engram_purge_combo.setMinimumHeight(32)
        self.engram_purge_button = QPushButton("Purge Engrams")
        self.engram_purge_button.setMinimumHeight(30)
        engram_row.addWidget(self.engram_purge_combo, 1)
        engram_row.addWidget(self.engram_purge_button)
        settings_layout.addLayout(engram_row)

        rag_title = QLabel("File RAG")
        rag_title.setObjectName("panelTitle")
        settings_layout.addWidget(rag_title)

        rag_subtitle = QLabel("Choose directories Celeste can search when answering questions about local files.")
        rag_subtitle.setWordWrap(True)
        rag_subtitle.setObjectName("panelSubtitle")
        settings_layout.addWidget(rag_subtitle)

        self.rag_dirs_list = QListWidget()
        self.rag_dirs_list.setMinimumHeight(150)
        settings_layout.addWidget(self.rag_dirs_list)

        rag_buttons = QGridLayout()
        rag_buttons.setHorizontalSpacing(10)
        rag_buttons.setVerticalSpacing(10)
        self.add_directory_button = QPushButton("Add Directory")
        self.remove_directory_button = QPushButton("Remove Directory")
        self.reindex_button = QPushButton("Reindex Files")
        self.build_deep_button = QPushButton("Build Deep Index")
        for button in (
            self.add_directory_button,
            self.remove_directory_button,
            self.reindex_button,
            self.build_deep_button,
        ):
            button.setMinimumHeight(30)
            button.setMinimumWidth(0)
        rag_buttons.addWidget(self.add_directory_button, 0, 0)
        rag_buttons.addWidget(self.remove_directory_button, 0, 1)
        rag_buttons.addWidget(self.reindex_button, 1, 0)
        rag_buttons.addWidget(self.build_deep_button, 1, 1)
        rag_buttons.setColumnStretch(0, 1)
        rag_buttons.setColumnStretch(1, 1)
        settings_layout.addLayout(rag_buttons)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        self.reload_button = QPushButton("Apply and Reload")
        self.refresh_models_button = QPushButton("Refresh Models")
        button_row.addWidget(self.reload_button)
        button_row.addWidget(self.refresh_models_button)
        settings_layout.addLayout(button_row)

        self.status_label = QLabel("Starting...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        settings_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        settings_layout.addWidget(self.progress_bar)
        settings_layout.addStretch(1)

        chat_card = QFrame()
        chat_card.setObjectName("chatCard")
        chat_layout = QVBoxLayout(chat_card)
        chat_layout.setContentsMargins(18, 18, 18, 18)
        chat_layout.setSpacing(12)

        chat_title = QLabel("Conversation")
        chat_title.setObjectName("panelTitle")
        chat_layout.addWidget(chat_title)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view, 1)

        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText("Type a prompt for Celeste...")
        self.input_box.setFixedHeight(120)
        chat_layout.addWidget(self.input_box)

        send_row = QHBoxLayout()
        send_row.setSpacing(10)
        self.send_button = QPushButton("Send")
        self.clear_button = QPushButton("Clear Transcript")
        send_row.addWidget(self.send_button)
        send_row.addWidget(self.clear_button)
        send_row.addStretch(1)
        chat_layout.addLayout(send_row)

        layout.addWidget(settings_card, 0)
        layout.addWidget(chat_card, 1)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        self.setCentralWidget(root)

        self.deep_index_dialog = QProgressDialog("Building deep library index...", "", 0, 100, self)
        self.deep_index_dialog.setWindowTitle("Building Deep Index")
        self.deep_index_dialog.setCancelButton(None)
        self.deep_index_dialog.setAutoClose(False)
        self.deep_index_dialog.setAutoReset(False)
        self.deep_index_dialog.setMinimumDuration(0)
        self.deep_index_dialog.setWindowModality(Qt.ApplicationModal)
        self.deep_index_dialog.hide()

        mono = QFont("DejaVu Sans Mono", 10)
        self.chat_view.setFont(mono)
        self.input_box.setFont(mono)

        self.reload_button.clicked.connect(self._apply_settings)
        self.refresh_models_button.clicked.connect(lambda: self.load_models_requested.emit())
        self.send_button.clicked.connect(self._send_message)
        self.clear_button.clicked.connect(self.chat_view.clear)
        self.add_directory_button.clicked.connect(self._choose_rag_directory)
        self.remove_directory_button.clicked.connect(self._remove_selected_rag_directory)
        self.reindex_button.clicked.connect(lambda: self.reindex_rag_requested.emit())
        self.build_deep_button.clicked.connect(self._start_deep_index_build)
        self.engram_purge_button.clicked.connect(self._purge_engram_memory)
        self.engram_auto_toggle.toggled.connect(self._set_engram_auto_prune)

        self._set_busy(True, "Starting Celeste...")
        self.setStyleSheet(
            """
            QWidget {
                background: #0f1419;
                color: #edf2f7;
                font-family: "DejaVu Sans";
                font-size: 14px;
            }
        QFrame#settingsCard, QFrame#chatCard {
            background: #162029;
            border: 1px solid #274050;
            border-radius: 16px;
        }
        QLabel#panelTitle {
            font-size: 18px;
            font-weight: 700;
            color: #a8ffcf;
        }
        QLabel#panelSubtitle, QLabel#statusLabel {
            color: #a9b7c6;
            font-size: 12px;
        }
        QTextBrowser, QPlainTextEdit, QComboBox, QSpinBox {
            background: #0c1116;
            border: 1px solid #2b3946;
            border-radius: 10px;
            padding: 4px 8px;
            min-height: 24px;
            font-size: 12px;
        }
        QPushButton {
            background: #1f7a5c;
            border: none;
            border-radius: 10px;
            padding: 5px 8px;
            font-weight: 600;
            font-size: 11px;
            min-width: 0px;
        }
        QPushButton:disabled {
            background: #33414d;
            color: #7f8b96;
        }
        QPushButton:hover:!disabled {
            background: #27956f;
        }
        QCheckBox {
            spacing: 6px;
            font-size: 12px;
        }
            """
        )

    def _build_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = ServiceWorker(self.config_path)
        self.worker.moveToThread(self.worker_thread)

        self.initialize_requested.connect(self.worker.initialize)
        self.load_models_requested.connect(self.worker.load_models)
        self.send_requested.connect(self.worker.send_message)
        self.reload_requested.connect(self.worker.reload_config)
        self.add_rag_directory_requested.connect(self.worker.add_rag_directory)
        self.remove_rag_directory_requested.connect(self.worker.remove_rag_directory)
        self.reindex_rag_requested.connect(self.worker.reindex_rag)
        self.build_deep_index_requested.connect(self.worker.build_deep_index)
        self.purge_engram_requested.connect(self.worker.purge_engram_memory)
        self.set_engram_auto_prune_requested.connect(self.worker.set_engram_auto_prune)
        self.shutdown_requested.connect(self.worker.shutdown)

        self.worker.initialized.connect(self._on_initialized)
        self.worker.models_ready.connect(self._on_models_ready)
        self.worker.reloaded.connect(self._on_reloaded)
        self.worker.rag_updated.connect(self._on_rag_updated)
        self.worker.memory_updated.connect(self._on_memory_updated)
        self.worker.reply_ready.connect(self._on_reply_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.status.connect(self._set_status)
        self.worker.progress.connect(self._on_progress)

        self.worker_thread.start()

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        self.send_button.setEnabled(not busy)
        self.input_box.setEnabled(not busy)
        self.reload_button.setEnabled(not busy)
        self.refresh_models_button.setEnabled(not busy)
        self.add_directory_button.setEnabled(not busy)
        self.remove_directory_button.setEnabled(not busy)
        self.reindex_button.setEnabled(not busy)
        self.build_deep_button.setEnabled(not busy)
        self.engram_auto_toggle.setEnabled(not busy)
        self.engram_purge_combo.setEnabled(not busy)
        self.engram_purge_button.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.tokens_spin.setEnabled(not busy)
        self.tts_toggle.setEnabled(not busy)
        self.memory_toggle.setEnabled(not busy)
        self.reflection_toggle.setEnabled(not busy)
        self.rag_dirs_list.setEnabled(not busy)
        if status:
            self.status_label.setText(status)
        if not busy:
            self._busy_reason = None
            self._hide_progress_ui()

    def _show_deep_index_progress(self, message: str, percent: int = 0) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(max(0, min(100, int(percent))))
        self.progress_bar.setFormat(f"{self.progress_bar.value()}%")
        self.deep_index_dialog.setLabelText(message)
        self.deep_index_dialog.setValue(max(0, min(100, int(percent))))
        if not self.deep_index_dialog.isVisible():
            self.deep_index_dialog.show()

    def _hide_progress_ui(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        if self.deep_index_dialog.isVisible():
            self.deep_index_dialog.hide()
        self.deep_index_dialog.setValue(0)

    @Slot(str)
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        if self._busy_reason == "deep_index":
            self._show_deep_index_progress(message, percent)

    @Slot(object)
    def _on_initialized(self, cfg: object) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else None
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system("Celeste backend is online.")
        self._set_busy(False, "Celeste ready.")

    @Slot(object)
    def _on_reloaded(self, cfg: object) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else None
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system("Settings applied. Backend reloaded.")
        self._set_busy(False, "Reload complete.")

    @Slot(list)
    def _on_models_ready(self, models: list) -> None:
        current = self._selected_model_path()
        name_counts: dict[str, int] = {}
        for model in models:
            base = os.path.basename(str(model))
            name_counts[base] = name_counts.get(base, 0) + 1
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models:
            path = str(model)
            label = self._model_label(path, duplicate=name_counts.get(os.path.basename(path), 0) > 1)
            self.model_combo.addItem(label, userData=path)
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(index, path, Qt.ToolTipRole)
        if current:
            self._set_selected_model(current)
        elif self.cfg is not None:
            self._set_selected_model(self.cfg.model_path)
        self.model_combo.blockSignals(False)

    @Slot(object, str)
    def _on_rag_updated(self, cfg: object, message: str) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else self.cfg
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system(message)
        status = "Deep index ready." if self._busy_reason == "deep_index" else "Library index updated."
        self._set_busy(False, status)

    @Slot(object, str)
    def _on_memory_updated(self, cfg: object, message: str) -> None:
        self.cfg = cfg if isinstance(cfg, AgentConfig) else self.cfg
        if self.cfg is not None:
            self._populate_from_config(self.cfg)
        self._append_system(message)
        self._set_busy(False, "Engram memory updated.")

    @Slot(str, str, str)
    def _on_reply_ready(self, answer: str, critique: str, improvements: str) -> None:
        self._append_chat("Celeste", answer, "#a8ffcf", split_sources=True)
        if critique.strip():
            self._append_chat("Critique", critique, "#ffd6a5")
        if improvements.strip():
            self._append_chat("Playbook", improvements, "#9fd3ff")
        self._set_busy(False, "Ready.")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._append_system(f"Error: {message}")
        self._set_busy(False, "Error.")
        QMessageBox.critical(self, "Celeste Error", message)

    def _populate_from_config(self, cfg: AgentConfig) -> None:
        self._set_selected_model(cfg.model_path)
        self.tts_toggle.setChecked(bool(cfg.tts_enabled))
        self.memory_toggle.setChecked(bool(cfg.use_chroma))
        reflection_cfg = cfg.reflection or {}
        self.reflection_toggle.setChecked(bool(reflection_cfg.get("enabled", False)))
        self.tokens_spin.setValue(int(cfg.max_new_tokens))
        memory_cfg = dict(cfg.memory or {})
        self.engram_auto_toggle.blockSignals(True)
        self.engram_auto_toggle.setChecked(bool(memory_cfg.get("engram_auto_prune", True)))
        self.engram_auto_toggle.blockSignals(False)
        self.rag_dirs_list.clear()
        for path in cfg.file_rag_dirs:
            self.rag_dirs_list.addItem(path)

    def _model_label(self, path: str, *, duplicate: bool = False) -> str:
        base = os.path.basename(path)
        if not duplicate:
            return base
        parent = os.path.basename(os.path.dirname(path)) or "models"
        return f"{base} [{parent}]"

    def _set_selected_model(self, model_path: str) -> None:
        index = self.model_combo.findData(model_path)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
            return
        label = self._model_label(model_path)
        self.model_combo.setEditText(label)
        self.model_combo.setToolTip(model_path)

    def _selected_model_path(self) -> str:
        data = self.model_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data
        return self.model_combo.currentText().strip()

    def _append_system(self, text: str) -> None:
        self.chat_view.append(
            f"<div style='margin: 6px 0; color: #8da2b5;'><i>{html.escape(text)}</i></div>"
        )

    def _split_sources_block(self, text: str) -> tuple[str, list[str]]:
        match = re.search(r"\n\s*Sources:\s*\n", text or "")
        if not match:
            return text, []
        body = (text or "")[: match.start()].rstrip()
        tail = (text or "")[match.end():]
        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        return body, lines

    def _append_chat(self, speaker: str, text: str, color: str, *, split_sources: bool = False) -> None:
        body_text = text
        source_lines: list[str] = []
        if split_sources:
            body_text, source_lines = self._split_sources_block(text)
        safe_text = html.escape(body_text).replace("\n", "<br>")
        safe_speaker = html.escape(speaker)
        sources_html = ""
        if source_lines:
            rendered_sources = "".join(
                f"<div style='margin: 4px 0;'>{html.escape(line)}</div>"
                for line in source_lines
            )
            sources_html = (
                "<div style='margin-top: 8px; background: #121922; border: 1px solid #314353; "
                "border-radius: 10px; padding: 10px;'>"
                "<div style='font-weight: 700; color: #9fd3ff; margin-bottom: 6px;'>Sources</div>"
                f"{rendered_sources}</div>"
            )
        self.chat_view.append(
            f"<div style='margin: 10px 0;'>"
            f"<div style='font-weight: 700; color: {color}; margin-bottom: 4px;'>{safe_speaker}</div>"
            f"<div style='background: #0c1116; border: 1px solid #25313d; border-radius: 10px; padding: 10px;'>"
            f"{safe_text}</div>"
            f"{sources_html}</div>"
        )

    def _send_message(self) -> None:
        if self.busy:
            return
        message = self.input_box.toPlainText().strip()
        if not message:
            return
        self.input_box.clear()
        self._append_chat("You", message, "#9fd3ff")
        self._set_busy(True, "Generating reply...")
        self.send_requested.emit(message)

    def _choose_rag_directory(self) -> None:
        if self.busy:
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose Directory for Celeste File Access")
        if not directory:
            return
        self._set_busy(True, f"Indexing {directory}...")
        self.add_rag_directory_requested.emit(directory)

    def _remove_selected_rag_directory(self) -> None:
        if self.busy:
            return
        item = self.rag_dirs_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Remove Directory", "Select a directory from the list first.")
            return
        directory = item.text().strip()
        if not directory:
            return
        self._set_busy(True, f"Removing {directory}...")
        self.remove_rag_directory_requested.emit(directory)

    def _apply_settings(self) -> None:
        if self.busy:
            return
        reflection = dict((self.cfg.reflection if self.cfg is not None else {}) or {})
        reflection["enabled"] = self.reflection_toggle.isChecked()
        memory_cfg = dict((self.cfg.memory if self.cfg is not None else {}) or {})
        memory_cfg["engram_auto_prune"] = self.engram_auto_toggle.isChecked()
        overrides = {
            "model_path": self._selected_model_path(),
            "tts_enabled": self.tts_toggle.isChecked(),
            "use_chroma": self.memory_toggle.isChecked(),
            "max_new_tokens": int(self.tokens_spin.value()),
            "reflection": reflection,
            "memory": memory_cfg,
        }
        self._set_busy(True, "Applying settings...")
        self.reload_requested.emit(overrides, True)

    def _set_engram_auto_prune(self, enabled: bool) -> None:
        if self.busy:
            return
        self._set_busy(True, "Updating Engram auto-purge...")
        self.set_engram_auto_prune_requested.emit(bool(enabled))

    def _purge_engram_memory(self) -> None:
        if self.busy:
            return
        seconds = self.engram_purge_combo.currentData()
        label = self.engram_purge_combo.currentText().strip() or "this Engram range"
        reply = QMessageBox.question(
            self,
            "Purge Engram Memory",
            f"Proceed with: {label}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._set_busy(True, "Purging Engram memory...")
        self.purge_engram_requested.emit(seconds)

    def _start_deep_index_build(self) -> None:
        if self.busy:
            return
        self._busy_reason = "deep_index"
        self._set_busy(True, "Building deep library index...")
        self._show_deep_index_progress("Building deep library index...", 0)
        self.build_deep_index_requested.emit()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            QMetaObject.invokeMethod(self.worker, "shutdown", Qt.BlockingQueuedConnection)
        except Exception:
            self.shutdown_requested.emit()
        self.worker_thread.quit()
        self.worker_thread.wait(5000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    icon_path = resource_path("assets", "celeste_icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    config_path = default_config_path()
    if not os.path.exists(config_path):
        from setup_wizard import ensure_config_with_wizard

        if not ensure_config_with_wizard(config_path, app=app):
            return 0
    window = CelesteWindow(config_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
