# -*- coding: utf-8 -*-
"""
PNMS 版本信息：区分「库程序版本」与「记忆文件（checkpoint）格式版本」。

- **库程序版本**：随功能迭代递增，仅升级 pip 包即可；不要求迁移磁盘上的 checkpoint。
- **记忆文件格式版本**：描述 meta.json / graph.db / .pt 等磁盘布局与字段语义；
  当该版本变更时，集成方可能需要迁移数据或同时升级调用代码。

二者独立维护：可能出现「库已 0.2.0，记忆格式仍为 1.0.0」的情况。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# 与 pyproject.toml [project].version 保持一致（发布时请同步修改两处）
LIBRARY_VERSION = "0.1.0"

# 记忆 checkpoint 布局版本（语义化：主版本不兼容时需在文档中说明迁移路径）
MEMORY_FORMAT_VERSION = "1.0.0"


def get_library_version() -> str:
    """返回当前安装的 PNMS 库程序版本字符串。"""
    return LIBRARY_VERSION


def get_memory_format_version() -> str:
    """返回当前库所读写的记忆文件格式版本字符串。"""
    return MEMORY_FORMAT_VERSION


def get_versions() -> Dict[str, str]:
    """
    一次性返回程序版本与记忆格式版本，便于日志与兼容性检查。

    Returns:
        ``{"library": "...", "memory_format": "..."}``
    """
    return {
        "library": LIBRARY_VERSION,
        "memory_format": MEMORY_FORMAT_VERSION,
    }


def peek_checkpoint_versions(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    仅读取 checkpoint 目录中的版本元数据，不加载完整模型与图到内存。

    用于集成方在调用 ``PNMS.load_concept_modules`` 之前判断是否需要迁移。

    Args:
        checkpoint_dir: ``concept_checkpoint_dir`` 对应目录（含 ``meta.json``、可选 ``graph.db``）。

    Returns:
        字典字段说明：
        - ``memory_format_in_meta``: ``meta.json`` 中的 ``memory_format_version``，缺省为 ``None``（旧 checkpoint）。
        - ``library_saved_in_meta``: 保存时写入的 ``pnms_library_version``，缺省为 ``None``。
        - ``memory_format_in_graph``: ``graph.db`` 内 ``pnms_meta`` 表中的格式版本，缺省为 ``None``。
        - ``current_library``: 当前库 ``LIBRARY_VERSION``。
        - ``current_memory_format``: 当前库 ``MEMORY_FORMAT_VERSION``。
    """
    root = Path(checkpoint_dir)
    out: Dict[str, Any] = {
        "memory_format_in_meta": None,
        "library_saved_in_meta": None,
        "memory_format_in_graph": None,
        "current_library": LIBRARY_VERSION,
        "current_memory_format": MEMORY_FORMAT_VERSION,
    }
    meta_path = root / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out["memory_format_in_meta"] = meta.get("memory_format_version")
            out["library_saved_in_meta"] = meta.get("pnms_library_version")
        except (json.JSONDecodeError, OSError):
            pass

    graph_db = root / "graph.db"
    if graph_db.is_file():
        try:
            conn = sqlite3.connect(str(graph_db))
            try:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='pnms_meta'"
                )
                if cur.fetchone():
                    cur = conn.execute(
                        "SELECT value FROM pnms_meta WHERE key = 'memory_format_version'"
                    )
                    row = cur.fetchone()
                    if row:
                        out["memory_format_in_graph"] = row[0]
            finally:
                conn.close()
        except sqlite3.Error:
            pass

    return out


def parse_semver(version: str) -> Tuple[int, int, int]:
    """
    解析 ``major.minor.patch`` 版本号。

    Raises:
        ValueError: 当版本字符串不符合三段纯数字语义化格式时抛出。
    """
    parts = version.strip().split(".")
    if len(parts) != 3:
        raise ValueError(f"invalid semver '{version}': expected major.minor.patch")
    try:
        major, minor, patch = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"invalid semver '{version}': each part must be integer") from e
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(f"invalid semver '{version}': negative number is not allowed")
    return major, minor, patch


def is_backward_compatible_checkpoint(
    checkpoint_memory_format_version: str,
    supported_memory_format_version: str,
) -> bool:
    """
    检查「只保证向下兼容，不保证向上兼容」规则。

    返回 True 的条件为：
    - checkpoint 版本 <= 当前（或调用方声明的）支持版本
    """
    ck = parse_semver(checkpoint_memory_format_version)
    sup = parse_semver(supported_memory_format_version)
    return ck <= sup
