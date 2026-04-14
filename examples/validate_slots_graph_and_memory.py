#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格实验：验证槽记忆、图扩展、以及回答是否引用记忆而非仅模型先验。
- 10+ 轮对话，确保进入 slots_and_graph 阶段，观察图扩展是否改变回答。
- content_to_remember 使用「结构化记忆摘要」而非用户原话，便于区分记忆 vs 先验。
- 针对二次方程等知识类问题，写入明确公式摘要，后续追问可检验是否从记忆复现。

运行：在项目根目录  python examples/validate_slots_graph_and_memory.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
from pnms import PNMS, PNMSConfig

OLLAMA_MODEL = "gpt-oss:20b"

# 冷启动：N0=3, N1=8 → 第 1~2 轮 pure_llm，第 3~7 轮 slots_only，第 8 轮起 slots_and_graph
COLD_START_N0 = 3
COLD_START_N1 = 8


def ollama_llm(query: str, context: str) -> str:
    try:
        from ollama import chat
    except ImportError:
        raise ImportError("请先安装 ollama：pip install ollama")
    system_content = context.strip() or "你是一个个人助手，请仅根据上述记忆与上下文回答，不要编造。"
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


def main():
    config = PNMSConfig(
        embed_dim=64,
        cold_start_n0=COLD_START_N0,
        cold_start_n1=COLD_START_N1,
        max_slots_per_user=100,
        retrieval_top_k=5,
        max_context_slots=8,
        graph_enabled=True,
        graph_max_neighbors_per_slot=3,
        graph_max_expanded_total=15,
        concept_enabled=False,
    )
    device = torch.device("cpu")
    pnms = PNMS(config=config, user_id="user_1", device=device)
    pnms.encoder.eval()

    # (用户 query, 写入记忆的结构化摘要；None 表示不写入新槽或沿用上一轮/自动)
    # 知识类写明确公式/事实，偏好类写简短结论，便于后续验证「是否引用了记忆」
    QUERIES_AND_MEMORY = [
        ("我喜欢用 Python 写算法。", "用户偏好：使用 Python 写算法。"),
        ("二次方程求根公式是什么？", "二次方程求根公式: x = (-b ± sqrt(b^2 - 4ac)) / (2a)，其中 a≠0。"),
        ("请再写一遍那个求根公式，我记到本子上。", "用户曾要求将求根公式写于本子；公式: x = (-b ± sqrt(b^2 - 4ac)) / (2a)。"),
        ("你记得我喜欢用什么语言吗？", None),  # 验证槽记忆：应召回「Python」
        ("刚才的公式再发我一次。", None),  # 验证槽记忆：应召回公式摘要
        ("我偏好简洁的答案。", "用户偏好：简洁的答案。"),
        ("我平时用 Linux。", "用户常用环境：Linux。"),
        ("我在做机器学习项目。", "用户当前在做：机器学习项目。"),
        ("总结一下你记得的关于我的事。", None),  # 验证多槽召回与摘要
        ("把求根公式再写一遍，只要公式不要解释。", None),  # 进入 slots_and_graph 后再次验证公式是否来自记忆
        ("我喜欢的编程语言是？", None),  # 图扩展后是否仍能召回 Python 偏好
        ("一元二次方程根的公式？", None),  # 换说法问公式，验证记忆键与检索
    ]

    print("=== 实验：验证槽记忆、图扩展、记忆 vs 先验 ===\n")
    print(f"冷启动: N0={COLD_START_N0}, N1={COLD_START_N1} → 第 {COLD_START_N1} 轮起进入 slots_and_graph\n")

    for i, (q, content_to_remember) in enumerate(QUERIES_AND_MEMORY):
        summary_before = pnms.get_state_summary()
        phase_before = summary_before["phase"]
        n_slots_before = summary_before["num_slots"]

        resp = pnms.handle_query(
            q,
            llm=ollama_llm,
            content_to_remember=content_to_remember,
            system_prompt="你是个人助手。请严格依据上面给出的「记忆」回答；若记忆中有公式或事实，请直接引用记忆中的表述。",
        )

        summary_after = pnms.get_state_summary()
        phase_after = summary_after["phase"]
        n_slots_after = summary_after["num_slots"]
        in_graph_phase = phase_after == "slots_and_graph"

        # 简要标注本轮在验证什么
        if content_to_remember is None and "公式" in q or "求根" in q:
            tag = "[验证: 公式是否来自记忆]"
        elif content_to_remember is None and ("语言" in q or "Python" in q or "喜欢" in q):
            tag = "[验证: 偏好槽记忆]"
        elif content_to_remember is None and "总结" in q:
            tag = "[验证: 多槽召回与摘要]"
        elif in_graph_phase and content_to_remember is None:
            tag = "[验证: 图扩展阶段检索]"
        else:
            tag = "[写入结构化记忆]"

        print(f"轮次 {i+1} | 阶段: {phase_after} | 槽数: {n_slots_before} → {n_slots_after} {tag}")
        print(f"  Q: {q}")
        if content_to_remember:
            print(f"  写入: {content_to_remember[:70]}{'...' if len(content_to_remember) > 70 else ''}")
        print(f"  A: {resp[:200]}{'...' if len(resp) > 200 else ''}")
        print()

    print("=== 实验结束 ===")
    print("最终状态:", pnms.get_state_summary())
    print("\n结论建议：检查第 4、5、10、11、12 轮回答是否出现「Python」「x = (-b ± sqrt(b^2 - 4ac)) / (2a)」等，以判断槽记忆与图扩展阶段是否生效；若模型未引用记忆而仅用先验，可考虑加强 system 提示或检索条数。")


if __name__ == "__main__":
    main()
