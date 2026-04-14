#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂演示：验证语义编码（SentenceEncoder）下的多主题记忆与换说法召回。
- 使用默认 PNMS（启用 sentence-transformers 时为语义编码，embed_dim=384）。
- 多主题：编程偏好、公式、环境、项目、阅读习惯等；换不同说法追问同一记忆，检验语义检索。
- 轮次 18+，覆盖 pure_llm → slots_only → slots_and_graph，并在图阶段做跨主题摘要。
运行：在项目根目录  python examples/demo_semantic_memory.py
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
COLD_START_N0 = 3
COLD_START_N1 = 8


def ollama_llm(query: str, context: str) -> str:
    try:
        from ollama import chat
    except ImportError:
        raise ImportError("请先安装 ollama：pip install ollama")
    system_content = context.strip() or "你是个人助手，请依据上述记忆回答；若记忆中有明确事实或公式，请直接引用。"
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
    # 不指定 embed_dim，使用默认；若启用 SentenceEncoder 则自动为 384
    config = PNMSConfig(
        cold_start_n0=COLD_START_N0,
        cold_start_n1=COLD_START_N1,
        max_slots_per_user=150,
        retrieval_top_k=6,
        max_context_slots=10,
        graph_enabled=True,
        graph_max_neighbors_per_slot=4,
        graph_max_expanded_total=18,
        concept_enabled=False,
    )
    device = torch.device("cpu")
    pnms = PNMS(config=config, user_id="user_1", device=device)
    if hasattr(pnms.encoder, "eval"):
        pnms.encoder.eval()

    encoder_name = type(pnms.encoder).__name__
    embed_dim = getattr(pnms.encoder, "embed_dim", pnms.config.embed_dim)
    print("=== 复杂演示：语义记忆与换说法召回 ===\n")
    print(f"编码器: {encoder_name} (embed_dim={embed_dim})")
    print(f"冷启动: N0={COLD_START_N0}, N1={COLD_START_N1}\n")

    # (用户 query, 写入的结构化记忆；None 表示不写入新内容，仅检索)
    # 设计：多主题 + 同一事实用不同说法追问，验证语义检索
    STEPS = [
        ("我喜欢用 Python 写算法，也用 TypeScript 写前端。", "用户偏好：用 Python 写算法、TypeScript 写前端。"),
        ("一元二次方程 ax²+bx+c=0 的求根公式是什么？", "一元二次方程求根公式: x = (-b ± sqrt(b^2 - 4ac)) / (2a)，a≠0。"),
        ("把这个公式再给我一遍，我记到本子。", "用户曾要求将求根公式记于本子；公式同上。"),
        ("你记得我常用什么语言写代码？", None),  # 换说法：应用召回 Python/TS
        ("刚才那个公式怎么写？", None),  # 换说法问公式
        ("我平时用 Linux 和 macOS。", "用户常用环境：Linux 与 macOS。"),
        ("我偏好简洁、带代码示例的答案。", "用户偏好：答案简洁且带代码示例。"),
        ("我在做机器学习项目，用 PyTorch。", "用户当前项目：机器学习，使用 PyTorch。"),
        ("我最近在看《深度学习》花书。", "用户近期阅读：《深度学习》（花书）。"),
        ("总结一下你记得的关于我的事。", None),  # 多槽摘要
        ("我跟你提过的技术偏好有哪些？", None),  # 跨主题：偏好类
        ("一元二次方程根的表达式？", None),  # 再次换说法问公式
        ("我用的开发环境是？", None),  # 换说法问 OS
        ("我在看的书是？", None),
        ("我做的项目和技术栈？", None),  # 项目+栈
        ("把求根公式再写一遍，只要式子。", None),
        ("用一句话概括我的编程和阅读情况。", None),  # 强概括
        ("二次方程求根公式？", None),  # 最简说法
        ("你还记得哪些关于我的偏好和环境？", None),
    ]

    for i, (q, content_to_remember) in enumerate(STEPS):
        summary_before = pnms.get_state_summary()
        phase = summary_before["phase"]
        n_before = summary_before["num_slots"]

        resp = pnms.handle_query(
            q,
            llm=ollama_llm,
            content_to_remember=content_to_remember,
            system_prompt="你是个人助手。请严格依据上文的「记忆」回答；若记忆中有公式或事实，请直接引用。",
        )

        summary_after = pnms.get_state_summary()
        n_after = summary_after["num_slots"]
        in_graph = "图" if summary_after["phase"] == "slots_and_graph" else ""

        tag = "[写入]" if content_to_remember else "[检索]"
        print(f"轮次 {i+1:2d} | {summary_after['phase']:18s} | 槽 {n_before}→{n_after} {in_graph} {tag}")
        print(f"  Q: {q[:72]}{'...' if len(q) > 72 else ''}")
        if content_to_remember:
            print(f"  记: {content_to_remember[:60]}{'...' if len(content_to_remember) > 60 else ''}")
        print(f"  A: {resp[:180]}{'...' if len(resp) > 180 else ''}")
        print()

    print("=== 演示结束 ===")
    print("最终状态:", pnms.get_state_summary())
    print("\n验证点：换说法（如「刚才那个公式」「一元二次方程根的表达式」「二次方程求根公式」）应都能召回同一公式记忆；「技术偏好」「开发环境」「在看的书」应分别命中对应槽。")

if __name__ == "__main__":
    main()
