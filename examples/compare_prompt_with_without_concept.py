#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同一问题在「概念未启用」与「概念启用」下，经 PNMS 后生成并发送给 LLM 的提示词对比，检查是否有压缩。
- 先积累约 20 槽并形成 2 个概念模块，再对同一 query 分别取 use_concept=False / True 的 context。
- 输出两段提示词、字符数、槽数，便于对比顺序与长度。
运行：在项目根目录  python examples/compare_prompt_with_without_concept.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
from pnms import PNMS, PNMSConfig, SimpleQueryEncoder
from pnms.concept import form_and_train_one_module

COLD_START_N0 = 3
COLD_START_N1 = 15
MIN_SLOTS_TO_FORM_CONCEPTS = 20


def mock_llm(query: str, context: str) -> str:
    """不调真实 LLM，仅返回占位，用于快速攒槽。"""
    return "ok"


def form_concept_modules_from_slots(pnms: PNMS) -> bool:
    """形成 2 个概念模块并做 Augment 对比学习训练。"""
    if len(pnms.concept_manager._modules) > 0:
        return False
    slots = getattr(pnms.store, "_slots", [])
    if len(slots) < MIN_SLOTS_TO_FORM_CONCEPTS:
        return False
    n = MIN_SLOTS_TO_FORM_CONCEPTS
    keys = torch.stack([s.key.to(pnms.device).float().flatten() for s in slots[:n]])
    c0 = keys[:10].mean(dim=0)
    c1 = keys[10:20].mean(dim=0)
    slot_ids_0 = [s.slot_id for s in slots[:10]]
    slot_keys_0 = [s.key for s in slots[:10]]
    slot_ids_1 = [s.slot_id for s in slots[10:20]]
    slot_keys_1 = [s.key for s in slots[10:20]]
    state = pnms.state.state
    form_and_train_one_module(
        pnms.concept_manager, "pref_env", c0, slot_ids_0, slot_keys_0,
        state, device=pnms.device, epochs=20, lr=1e-3,
    )
    form_and_train_one_module(
        pnms.concept_manager, "fact_formula", c1, slot_ids_1, slot_keys_1,
        state, device=pnms.device, epochs=20, lr=1e-3,
    )
    return True


def main():
    config = PNMSConfig(
        cold_start_n0=COLD_START_N0,
        cold_start_n1=COLD_START_N1,
        max_slots_per_user=80,
        retrieval_top_k=6,
        max_context_slots=10,
        graph_enabled=True,
        graph_max_neighbors_per_slot=4,
        graph_max_expanded_total=18,
        concept_enabled=True,
        concept_top_m=2,
    )
    device = torch.device("cpu")
    # 用简单编码器加快对比；若需语义编码可改为 PNMS(config=config, ...)
    encoder = SimpleQueryEncoder(config.embed_dim or 64, vocab_size=10000)
    pnms = PNMS(config=config, user_id="user_1", encoder=encoder, device=device)
    if hasattr(pnms.encoder, "eval"):
        pnms.encoder.eval()

    # 与 validate_concept_phase 相同的前 20 条，用于攒槽（不调真实 LLM）
    STEPS = [
        ("我喜欢用 Python 写算法。", "用户偏好：使用 Python 写算法。"),
        ("二次方程求根公式是什么？", "二次方程求根公式: x = (-b ± sqrt(b^2 - 4ac)) / (2a)，a≠0。"),
        ("把这个公式再给我一遍。", "用户曾要求给出求根公式；公式同上。"),
        ("你记得我喜欢什么语言？", None),
        ("刚才的公式怎么写？", None),
        ("我偏好简洁答案。", "用户偏好：简洁的答案。"),
        ("我平时用 Linux。", "用户常用环境：Linux。"),
        ("我在做机器学习项目，用 PyTorch。", "用户当前项目：机器学习，使用 PyTorch。"),
        ("我最近在看《深度学习》花书。", "用户近期阅读：《深度学习》（花书）。"),
        ("总结一下你记得的关于我的事。", None),
        ("我用的开发环境？", None),
        ("一元二次方程根的表达式？", None),
        ("我喜欢的编程语言是？", None),
        ("把求根公式再写一遍，只要式子。", None),
        ("我做的项目和技术栈？", None),
        ("技术偏好有哪些？", None),
        ("二次方程求根公式？", None),
        ("在看的书是？", None),
        ("常用什么写算法？", None),
        ("求根公式再发一次。", None),
    ]

    for i, (q, content_to_remember) in enumerate(STEPS):
        pnms.handle_query(q, llm=mock_llm, content_to_remember=content_to_remember, system_prompt="你是个人助手。")
        if len(pnms.store._slots) >= MIN_SLOTS_TO_FORM_CONCEPTS:
            form_concept_modules_from_slots(pnms)
            break
    # 若未满 20 槽，再补几轮写入（上限 5 轮防死循环）
    for _ in range(5):
        if len(pnms.store._slots) >= MIN_SLOTS_TO_FORM_CONCEPTS:
            break
        pnms.handle_query("补充记忆。", llm=mock_llm, content_to_remember="用户说：补充记忆。", system_prompt="你是个人助手。")
    if len(pnms.store._slots) >= MIN_SLOTS_TO_FORM_CONCEPTS:
        form_concept_modules_from_slots(pnms)

    system_prompt = "你是个人助手。请严格依据上文的「记忆」回答；若记忆中有公式或事实请直接引用。"
    query = "你记得我喜欢用什么语言吗？"

    context_no_concept, n_slots_no = pnms.get_context_for_query(query, system_prompt=system_prompt, use_concept=False)
    context_with_concept, n_slots_with = pnms.get_context_for_query(query, system_prompt=system_prompt, use_concept=True)

    def safe_len(s: str) -> int:
        return len(s) if s else 0

    len_no = safe_len(context_no_concept)
    len_with = safe_len(context_with_concept)

    print("=== 同一问题在概念启用前后的提示词对比 ===\n")
    print(f"问题: {query}\n")
    print("--- 概念未启用时发送给 LLM 的提示词 (use_concept=False) ---")
    print(f"槽数: {n_slots_no} | 字符数: {len_no}")
    print(context_no_concept)
    print("\n--- 概念启用后经 PNMS 生成的提示词 (use_concept=True) ---")
    print(f"槽数: {n_slots_with} | 字符数: {len_with}")
    print(context_with_concept)
    print("\n--- 对比 ---")
    print(f"槽数: {n_slots_no} → {n_slots_with} (相同则仅重排，未删槽)")
    print(f"字符数: {len_no} → {len_with} (差值: {len_with - len_no})")
    if len_no == len_with and n_slots_no == n_slots_with:
        print("结论: 本实现中概念阶段仅对槽做「重排」(Augment)，不减少槽数或截断内容，故无字符级压缩；压缩体现在更相关的记忆优先进入固定 max_context_slots 预算。")
    else:
        print("结论: 槽数或字符数有变化，可能因概念重排后截断顺序不同导致入选槽不同。")

if __name__ == "__main__":
    main()
