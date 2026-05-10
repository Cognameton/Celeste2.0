from __future__ import annotations

import os
import sys


def runtime_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_root() -> str:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.abspath(str(meipass))
    return runtime_root()


def resource_path(*parts: str) -> str:
    return os.path.join(resource_root(), *parts)


_APP_NAME = "Celeste2"


def _user_data_root() -> str:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, _APP_NAME)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, _APP_NAME)


def default_config_path() -> str:
    if not getattr(sys, "frozen", False):
        return os.path.join(runtime_root(), "config.yaml")
    return os.path.join(_user_data_root(), "config.yaml")


def default_self_path() -> str:
    if not getattr(sys, "frozen", False):
        return os.path.join(runtime_root(), "self")
    return os.path.join(_user_data_root(), "self")
