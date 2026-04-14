# -*- coding: utf-8 -*-
"""
记忆图 G = (V, E)（文档 §8）。
节点与记忆槽一一对应，边为共现关系，边权 w_ij 可更新与衰减；在线仅做 1 跳扩展。
持久化：使用 SQLite 单文件（graph.db），通过索引支持按节点快速查邻居，事务安全。
注：槽的向量检索（top-k 相似）在 MemoryStore，若需向量库级加速可后续接 sqlite-vec 等扩展。
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

from .slot import MemorySlot
from .exceptions import PersistenceError
from .versioning import MEMORY_FORMAT_VERSION

_EDGES_TABLE = "edges"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    slot_i TEXT NOT NULL,
    slot_j TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (slot_i, slot_j)
);
CREATE INDEX IF NOT EXISTS idx_edges_i ON edges(slot_i);
CREATE TABLE IF NOT EXISTS pnms_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


logger = logging.getLogger("pnms.graph")


class MemoryGraph:
    """
    记忆图（文档 §8.1）：节点 V 对应槽 ID，边 E 为 (i, j, w_ij)。
    在线：仅对给定槽集合做 1 跳邻居扩展（§8.3、§8.4）。
    边权更新：共现时 +δ；可选归一化/衰减（§8.2）。
    """

    def __init__(self, cooccur_delta: float = 0.1, edge_decay: float = 0.99, max_edge_weight: float = 10.0):
        """
        Args:
            cooccur_delta: 共现时边权增加量 δ。
            edge_decay: 边权衰减 λ（调用 decay_edges 时使用）。
            max_edge_weight: 边权上界，避免爆炸。
        """
        self.cooccur_delta = cooccur_delta
        self.edge_decay = edge_decay
        self.max_edge_weight = max_edge_weight
        # 邻接：slot_id -> List[(neighbor_id, weight)]
        self._adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        # 反向邻接，便于按 (i,j) 更新
        self._edges: Dict[Tuple[str, str], float] = {}

    def record_cooccurrence(self, slot_ids: List[str]) -> None:
        """
        记录本轮共同激活的槽对，更新边权（文档 §8.2）。
        w_ij^{t+1} = w_ij^t + δ，并做上界截断。
        """
        if len(slot_ids) < 2:
            return
        seen: Set[Tuple[str, str]] = set()
        for i in slot_ids:
            for j in slot_ids:
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                self._edges[pair] = min(
                    self._edges.get(pair, 0.0) + self.cooccur_delta,
                    self.max_edge_weight,
                )
        self._rebuild_adj()

    def _rebuild_adj(self) -> None:
        """从 _edges 重建邻接表，便于按节点查邻居。"""
        self._adj = defaultdict(list)
        for (i, j), w in self._edges.items():
            self._adj[i].append((j, w))
            self._adj[j].append((i, w))

    def get_neighbors(
        self,
        slot_id: str,
        top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        取 slot_id 的 1 跳邻居，按边权降序取 top_n（文档 §8.4）。
        返回 [(neighbor_id, weight), ...]。
        """
        if slot_id not in self._adj:
            return []
        neighbors = sorted(self._adj[slot_id], key=lambda x: -x[1])
        return neighbors[:top_n]

    def expand_slots(
        self,
        slot_ids: List[str],
        slot_by_id: Dict[str, MemorySlot],
        max_neighbors_per_slot: int,
        max_total: int,
    ) -> List[str]:
        """
        图扩展：对 slot_ids 中每个槽取 1 跳邻居，去重后合并，总数为 max_total 以内（文档 §8.4）。
        返回扩展后的槽 ID 列表（含原 slot_ids，去重、按“先原槽再邻居”顺序，截断）。
        """
        out: List[str] = list(slot_ids)
        seen: Set[str] = set(slot_ids)
        for sid in slot_ids:
            for nid, _ in self.get_neighbors(sid, top_n=max_neighbors_per_slot):
                if nid in seen or nid not in slot_by_id:
                    continue
                seen.add(nid)
                out.append(nid)
                if len(out) >= max_total:
                    return out[:max_total]
        return out[:max_total]

    def decay_edges(self) -> None:
        """全局边权衰减 w_ij <- λ * w_ij（文档 §8.2），弱边可后续剪枝。"""
        self._edges = {k: v * self.edge_decay for k, v in self._edges.items()}
        self._rebuild_adj()

    def remove_slot(self, slot_id: str) -> None:
        """删除某槽相关的所有边（槽被淘汰时调用）。"""
        to_remove = [k for k in self._edges if slot_id in k]
        for k in to_remove:
            del self._edges[k]
        self._rebuild_adj()

    def merge_edges_from_graph_db(
        self,
        path: Union[str, Path],
        slot_ids: Set[str],
        *,
        combine: str = "max",
    ) -> int:
        """
        从另一份 checkpoint 的 ``graph.db`` 合并边到当前图。
        仅保留两端槽 ID 均存在于 ``slot_ids`` 中的边（合并后槽 ID 可能已变化，仅当 ID 仍存在时才能合并边）。

        ``combine``: ``max`` 取较大边权；``sum`` 相加后截断到 ``max_edge_weight``。
        """
        path = Path(path)
        if not path.is_file() or path.suffix.lower() != ".db":
            return 0
        merged = 0
        try:
            conn = sqlite3.connect(str(path))
            try:
                cur = conn.execute("SELECT slot_i, slot_j, weight FROM edges")
                for row in cur:
                    i, j = str(row[0]), str(row[1])
                    w = float(row[2])
                    if i not in slot_ids or j not in slot_ids:
                        continue
                    pair = (min(i, j), max(i, j))
                    if combine == "sum":
                        nw = min(
                            self.max_edge_weight,
                            self._edges.get(pair, 0.0) + w,
                        )
                    else:
                        nw = max(self._edges.get(pair, 0.0), w)
                    self._edges[pair] = nw
                    merged += 1
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning("merge_edges_from_graph_db: skip %s: %s", path, e)
            return 0
        self._rebuild_adj()
        return merged

    def save(self, path: Union[str, Path]) -> None:
        """
        将图边表保存到 SQLite（文档 §8.2、§13.2）。
        path 为 .db 文件路径；单文件、索引支持按节点快速查邻居，事务安全。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(path))
            try:
                conn.executescript(_SCHEMA)
                conn.execute("DELETE FROM edges")
                conn.executemany(
                    "INSERT OR REPLACE INTO edges (slot_i, slot_j, weight) VALUES (?, ?, ?)",
                    [(i, j, float(w)) for (i, j), w in self._edges.items()],
                )
                conn.execute(
                    "INSERT OR REPLACE INTO pnms_meta (key, value) VALUES ('memory_format_version', ?)",
                    (MEMORY_FORMAT_VERSION,),
                )
                conn.commit()
                logger.debug("saved %d edges into sqlite graph at %s", len(self._edges), path)
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.error("failed to save graph to sqlite %s: %s", path, e)
            raise PersistenceError(f"failed to save graph to sqlite: {path}") from e

    def load(self, path: Union[str, Path]) -> None:
        """
        从磁盘加载图边表，覆盖当前 _edges 并重建邻接表。
        仅支持从 SQLite 数据库文件（.db）读取。
        """
        path = Path(path)
        if not path.exists():
            return
        self._edges.clear()
        if path.suffix.lower() != ".db":
            raise PersistenceError(f"graph persistence now only supports SQLite .db files, got: {path}")
        try:
            conn = sqlite3.connect(str(path))
            try:
                cur = conn.execute("SELECT slot_i, slot_j, weight FROM edges")
                for row in cur:
                    i, j, w = row[0], row[1], float(row[2])
                    pair = (min(i, j), max(i, j))
                    self._edges[pair] = w
                try:
                    cur = conn.execute(
                        "SELECT value FROM pnms_meta WHERE key = 'memory_format_version'"
                    )
                    row = cur.fetchone()
                    if row and row[0] != MEMORY_FORMAT_VERSION:
                        logger.warning(
                            "graph.db memory_format_version=%s != current %s; "
                            "若边表异常请查阅 PNMS 迁移说明",
                            row[0],
                            MEMORY_FORMAT_VERSION,
                        )
                except sqlite3.OperationalError:
                    pass
            finally:
                conn.close()
            self._rebuild_adj()
            logger.info("loaded %d edges from sqlite graph %s", len(self._edges), path)
        except sqlite3.Error as e:
            logger.error("failed to load graph from sqlite %s: %s", path, e)
            raise PersistenceError(f"failed to load graph from sqlite: {path}") from e
