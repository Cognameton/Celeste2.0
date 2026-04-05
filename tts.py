"""
tts.py — minimal synchronous text-to-speech helpers for Celeste.

This module intentionally avoids background workers, persistent Piper sessions,
and streaming chunk assembly. A reply is synthesized and played in one
foreground call so the TTS flow stays predictable.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Optional

if os.name == "nt":  # pragma: no cover - Windows-only runtime path
    import winsound


class TTSManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled: bool = bool(getattr(cfg, "tts_enabled", False))
        self.backend: str = (getattr(cfg, "tts_backend", "pyttsx3") or "pyttsx3").lower()
        self._engine = None

        if not self.enabled:
            return

        if self.backend in ("pyttsx3", "espeak"):
            self._init_pyttsx3()
        elif self.backend == "piper":
            self._init_piper()
        else:
            print(f"[tts] Unknown backend '{self.backend}'. TTS disabled.")
            self.enabled = False

    def _init_pyttsx3(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            print(f"[tts] pyttsx3 unavailable ({exc}). TTS disabled.")
            self.enabled = False
            return

        try:
            engine = pyttsx3.init()
            voice_pref: Optional[str] = getattr(self.cfg, "tts_voice", None)
            rate: Optional[int] = getattr(self.cfg, "tts_rate", None)
            volume: Optional[float] = getattr(self.cfg, "tts_volume", None)

            if voice_pref:
                voice_pref_low = voice_pref.lower()
                selected = None
                for voice in engine.getProperty("voices"):
                    ident = f"{voice.id}".lower()
                    name = f"{voice.name}".lower()
                    if voice_pref_low in ident or voice_pref_low in name:
                        selected = voice.id
                        break
                if selected:
                    engine.setProperty("voice", selected)

            if isinstance(rate, int) and rate > 0:
                engine.setProperty("rate", rate)
            if isinstance(volume, (float, int)) and 0.0 <= float(volume) <= 1.0:
                engine.setProperty("volume", float(volume))

            self._engine = engine
            self.backend = "pyttsx3"
        except Exception as exc:
            print(f"[tts] pyttsx3 initialization failed ({exc}). TTS disabled.")
            self.enabled = False

    def _init_piper(self) -> None:
        exe = getattr(self.cfg, "tts_piper_executable", "piper") or "piper"
        model_path = getattr(self.cfg, "tts_piper_model", "") or ""
        config_path = getattr(self.cfg, "tts_piper_config", "") or ""
        speaker = getattr(self.cfg, "tts_piper_speaker", None)

        resolved_exe = shutil.which(exe)
        if not resolved_exe:
            print(f"[tts] Piper executable '{exe}' not found. TTS disabled.")
            self.enabled = False
            return
        if not model_path or not os.path.exists(model_path):
            print("[tts] Piper model path missing or invalid. TTS disabled.")
            self.enabled = False
            return
        if config_path and not os.path.exists(config_path):
            print("[tts] Piper config path invalid. TTS disabled.")
            self.enabled = False
            return

        self._piper_executable = resolved_exe
        self._piper_model = model_path
        self._piper_config = config_path or None
        self._piper_speaker = speaker
        self._piper_player = shutil.which("aplay")
        self._piper_args = self._resolve_piper_args()
        self._piper_env = self._build_piper_env(resolved_exe)

        if not self._supports_piper_cli():
            print(f"[tts] Executable '{resolved_exe}' is not a compatible Piper CLI binary. TTS disabled.")
            self.enabled = False
            return

        self.backend = "piper"

    def _resolve_piper_args(self) -> list[str]:
        presets = getattr(self.cfg, "tts_presets", {}) or {}
        preset_name = getattr(self.cfg, "tts_default_preset", None)
        extra_args = getattr(self.cfg, "tts_piper_args", "") or ""
        raw_args = presets.get(preset_name, "") if preset_name and isinstance(presets, dict) else ""
        if not raw_args:
            raw_args = extra_args
        if not raw_args:
            return []
        try:
            return shlex.split(raw_args)
        except Exception:
            return raw_args.split()

    def _build_piper_env(self, resolved_exe: str) -> dict[str, str]:
        env = os.environ.copy()
        lib_dir = os.path.dirname(resolved_exe)
        if os.name == "nt":
            existing_path = env.get("PATH", "")
            path_parts = existing_path.split(os.pathsep) if existing_path else []
            if lib_dir not in path_parts:
                env["PATH"] = lib_dir + (os.pathsep + existing_path if existing_path else "")
            return env
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing_ld}" if existing_ld else lib_dir

        preloads: list[str] = []
        try:
            for entry in os.listdir(lib_dir):
                if entry.startswith("libpiper_phonemize.so") or entry.startswith("libespeak-ng.so"):
                    preloads.append(os.path.join(lib_dir, entry))
        except Exception:
            return env

        if preloads:
            existing_preload = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = ":".join(preloads + ([existing_preload] if existing_preload else []))
        return env

    def _supports_piper_cli(self) -> bool:
        try:
            res = subprocess.run(
                [self._piper_executable, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
                env=self._piper_env,
                **self._piper_subprocess_kwargs(),
            )
        except Exception:
            return False
        help_text = (res.stdout or "") + (res.stderr or "")
        return "--model" in help_text or ("piper" in help_text.lower() and "usage" in help_text.lower())

    def _piper_subprocess_kwargs(self) -> dict:
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo
        return kwargs

    def speak(self, text: str) -> None:
        if not self.enabled:
            return
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return

        if self.backend == "pyttsx3":
            self._speak_pyttsx3(cleaned)
        elif self.backend == "piper":
            self._speak_piper(cleaned)

    def speak_async(self, text: str) -> None:
        self.speak(text)

    def start_stream(self) -> int:
        return 0

    def stream_text(self, stream_id: int, text: str) -> None:
        return

    def finish_stream(self, stream_id: int) -> None:
        return

    def shutdown(self) -> None:
        return

    def _speak_pyttsx3(self, text: str) -> None:
        if not self._engine:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            print(f"[tts] pyttsx3 playback failed ({exc}). Disabling TTS.")
            self.enabled = False

    def _speak_piper(self, text: str) -> None:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="celeste-tts-", suffix=".wav")
            os.close(fd)

            cmd = [
                self._piper_executable,
                "--model",
                self._piper_model,
                "--output_file",
                tmp_path,
            ]
            if self._piper_config:
                cmd.extend(["--config", self._piper_config])
            if self._piper_speaker:
                cmd.extend(["--speaker", str(self._piper_speaker)])
            cmd.extend(self._piper_args)

            subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._piper_env,
                **self._piper_subprocess_kwargs(),
            )

            if os.name == "nt":
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            elif self._piper_player:
                subprocess.run(
                    [self._piper_player, "-q", tmp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                print(f"[tts] Audio rendered to {tmp_path}")
                tmp_path = None
        except Exception as exc:
            print(f"[tts] Piper playback failed ({exc}). Disabling TTS.")
            self.enabled = False
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
