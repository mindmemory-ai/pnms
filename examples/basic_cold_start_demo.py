#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNMS 基础演示：多轮对话与冷启动阶段观察。
对应文档 §10 查询处理、§12 冷启动。

运行：在项目根目录执行  python examples/basic_cold_start_demo.py
或：cd examples && python basic_cold_start_demo.py
"""

import sys
from pathlib import Path

# 保证从 examples/ 或项目根运行都能正确 import pnms
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
from pnms import PNMS, PNMSConfig

OLLAMA_MODEL = "gpt-oss:20b"


def ollama_llm(query: str, context: str) -> str:
    try:
        from ollama import chat
    except ImportError:
        raise ImportError("请先安装 ollama：pip install ollama")
    system_content = context.strip() or "你是一个个人助手，请根据记忆回答。"
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
        cold_start_n0=3,
        cold_start_n1=8,
        max_slots_per_user=50,
        retrieval_top_k=3,
        max_context_slots=5,
        graph_enabled=True,
        concept_enabled=False,
    )
    device = torch.device("cpu")
    pnms = PNMS(config=config, user_id="user_1", device=device)
    pnms.encoder.eval()

    queries = [
        "我喜欢用 Python 写算法。",
        "二次方程求根公式是什么？",
        "请再写一遍那个求根公式，我记到本子上。",
        "你记得我喜欢用什么语言吗？",
        "刚才的公式再发我一次。",
        "我偏好简洁的答案。",
        "总结一下你记得的关于我的事。",
    ]

    print("=== PNMS 基础演示：冷启动与多轮对话 ===\n")
    for i, q in enumerate(queries):
        resp = pnms.handle_query(
            q,
            llm=ollama_llm,
            content_to_remember=q + " [用户表达]",
            system_prompt="你是一个个人助手，请根据记忆回答。",
        )
        summary = pnms.get_state_summary()
        print(f"轮次 {i+1} | 阶段: {summary['phase']} | 槽数: {summary['num_slots']}")
        print(f"  Q: {q[:60]}...")
        print(f"  A: {resp[:120]}...")
        print()
    print("=== 演示结束 ===")
    print("最终状态:", pnms.get_state_summary())


if __name__ == "__main__":
    main()
