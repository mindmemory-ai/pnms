#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证概念阶段：增加对话轮次、开启 concept_enabled，在槽数足够时形成概念模块并检验检索与回答。
- 约 30 轮对话，cold_start_n1=15，第 15 轮起进入 slots_and_graph。
- 第 21 轮后根据当前槽做简单“聚类”并注册概念模块，后续轮次应走概念 Augment 路径。
- 检查：概念阶段是否启用、回答是否仍正确引用记忆。
运行：在项目根目录  python examples/validate_concept_phase.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
from pnms import PNMS, PNMSConfig
from pnms.concept import form_and_train_one_module

OLLAMA_MODEL = "gpt-oss:20b"
COLD_START_N0 = 3
COLD_START_N1 = 15  # 第 15 轮起 slots_and_graph，便于尽早进入图+概念
MIN_SLOTS_TO_FORM_CONCEPTS = 20  # 槽数达到此值后形成概念模块（演示用）


def ollama_llm(query: str, context: str) -> str:
    try:
        from ollama import chat
    except ImportError:
        raise ImportError("请先安装 ollama：pip install ollama")
    system_content = context.strip() or "你是个人助手，请依据上述记忆回答。"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]
    response = chat(model=OLLAMA_MODEL, messages=messages)
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content or ""
    if isinstance(response, dict) and "message" in response and "content" in response["message"]:
        return response["message"]["content"] or ""
    return str(response)


def form_concept_modules_from_slots(pnms: PNMS) -> bool:
    """
    用当前记忆槽形成 2 个概念模块并做 Augment 对比学习训练（文档 §9.4、§9.5）。
    前 10 槽为「偏好/环境」簇，接下来 10 槽为「公式/项目」簇。
    仅当槽数 >= MIN_SLOTS_TO_FORM_CONCEPTS 且尚未注册过模块时执行一次。
    返回是否本次新注册并训练了模块。
    """
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
        concept_enabled=True,  # 开启概念阶段
        concept_top_m=2,
    )
    device = torch.device("cpu")
    pnms = PNMS(config=config, user_id="user_1", device=device)
    if hasattr(pnms.encoder, "eval"):
        pnms.encoder.eval()

    # 30 轮：前 20+ 轮积累槽并进入 slots_and_graph，之后形成概念，最后若干轮验证概念阶段
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
        ("用一句话概括我的偏好和环境。", None),  # 第 21 轮：此处之后会形成概念
        ("你记得我喜欢用什么语言吗？", None),  # 概念阶段验证：偏好
        ("刚才的公式再发我一次。", None),     # 概念阶段验证：公式
        ("总结你记得的关于我的事。", None),
        ("我喜欢的编程语言？", None),
        ("一元二次方程根的公式？", None),
        ("开发环境和项目？", None),
        ("把求根公式写一遍。", None),
        ("技术偏好和阅读？", None),
        ("二次方程求根公式？", None),
    ]

    print("=== 验证概念阶段 ===\n")
    print(f"冷启动: N0={COLD_START_N0}, N1={COLD_START_N1} → 第 {COLD_START_N1} 轮起 slots_and_graph")
    print(f"概念: concept_enabled=True，槽数>={MIN_SLOTS_TO_FORM_CONCEPTS} 时注册 2 个概念模块\n")

    concept_formed = False
    for i, (q, content_to_remember) in enumerate(STEPS):
        summary_before = pnms.get_state_summary()
        phase = summary_before["phase"]
        n_before = summary_before["num_slots"]
        n_concept_before = len(pnms.concept_manager._modules)

        # 每轮结束后检查是否达到形成概念的条件（在 handle_query 之后槽数会更新）
        resp = pnms.handle_query(
            q,
            llm=ollama_llm,
            content_to_remember=content_to_remember,
            system_prompt="你是个人助手。请严格依据上文的「记忆」回答；若记忆中有公式或事实请直接引用。",
        )

        summary_after = pnms.get_state_summary()
        n_after = summary_after["num_slots"]
        n_concept_after = len(pnms.concept_manager._modules)
        if not concept_formed and n_after >= MIN_SLOTS_TO_FORM_CONCEPTS:
            concept_formed = form_concept_modules_from_slots(pnms)
            n_concept_after = len(pnms.concept_manager._modules)
            if concept_formed:
                print(f"  >>> 已形成概念模块数: {n_concept_after} (pref_env, fact_formula)\n")

        in_graph = phase == "slots_and_graph"
        in_concept = in_graph and n_concept_after > 0
        phase_tag = "图" if in_graph else ""
        concept_tag = "概念" if in_concept else ""

        tag = "[写入]" if content_to_remember else "[检索]"
        print(f"轮次 {i+1:2d} | {summary_after['phase']:18s} | 槽 {n_before}→{n_after} 概念模块 {n_concept_after} {phase_tag}{concept_tag} {tag}")
        print(f"  Q: {q[:70]}{'...' if len(q) > 70 else ''}")
        if content_to_remember:
            print(f"  记: {content_to_remember[:55]}{'...' if len(content_to_remember) > 55 else ''}")
        print(f"  A: {resp[:160]}{'...' if len(resp) > 160 else ''}")
        print()

    print("=== 实验结束 ===")
    print("最终状态:", pnms.get_state_summary())
    print("概念模块数:", len(pnms.concept_manager._modules))
    print("\n预期：轮次 22 起应出现「概念」标记且回答仍正确引用偏好与公式记忆；若概念 Augment 生效，检索到的槽会经概念重排后送入 LLM。")


if __name__ == "__main__":
    main()
