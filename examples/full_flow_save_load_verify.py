#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流程：大样本训练 → 保存概念模型与图（graph.db）→ 加载模型与图 → 推理验证。

阶段一：约 70 轮多样化对话，进入 slots_and_graph 后触发概念形成，保存概念模块与图边表到磁盘。
阶段二：新建 PNMS、从磁盘加载概念模块与图（graph.db），用相同对话复现槽数据后做推理；
       验证概念/图加载、对比「启用/不启用概念」的检索结果。

运行（项目根目录）：python examples/full_flow_save_load_verify.py
使用 SimpleQueryEncoder 以加快执行；可改为 SentenceEncoder 以验证语义检索。
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch
from pnms import PNMS, PNMSConfig, SimpleQueryEncoder

# 概念模型保存目录（示例用，可改为持久路径）
CONCEPT_CKPT_DIR = Path(__file__).resolve().parent / "out_concept_ckpt"
# 冷启动与概念形成：尽早进入 slots_and_graph 并在约 25 轮尝试形成概念
COLD_START_N0 = 3
COLD_START_N1 = 18
CONCEPT_MIN_CLUSTER = 15
CONCEPT_FORMATION_INTERVAL = 25
TOTAL_ROUNDS = 70


def mock_llm(query: str, context: str) -> str:
    """不调真实 LLM，仅返回固定回复，用于快速攒槽与复现。"""
    return "已记录。"


def build_large_steps():
    """构造 70+ 轮多样化 (query, content_to_remember)，覆盖数学、偏好、环境、阅读等。"""
    steps = [
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
        ("用一句话概括我的偏好和环境。", None),
        ("你记得我喜欢用什么语言吗？", None),
        ("刚才的公式再发我一次。", None),
        ("总结你记得的关于我的事。", None),
        ("我喜欢的编程语言？", None),
        ("一元二次方程根的公式？", None),
        ("开发环境和项目？", None),
        ("把求根公式写一遍。", None),
        ("技术偏好和阅读？", None),
        ("二次方程求根公式？", None),
    ]
    # 再补一批，确保超过 30 轮并有多样性
    extra = [
        ("牛顿第二定律的公式？", "牛顿第二定律: F = ma。"),
        ("欧拉公式 e^(iπ)+1=0 是什么？", "欧拉公式: e^(iπ) + 1 = 0。"),
        ("勾股定理怎么表示？", "勾股定理: a² + b² = c²。"),
        ("我偏好用 Markdown 记笔记。", "用户偏好：用 Markdown 记笔记。"),
        ("我常用 VS Code。", "用户常用编辑器：VS Code。"),
        ("我在学线性代数。", "用户在学习：线性代数。"),
        ("我更喜欢矩阵运算的直观推导。", "用户偏好：矩阵运算的直观推导。"),
        ("SVD 分解是什么？", "SVD: 矩阵 A = U Σ V^T，用于降维与推荐。"),
        ("特征值分解和 SVD 的区别？", "特征值分解针对方阵；SVD 适用于任意矩阵。"),
        ("我平时用 conda 管理环境。", "用户环境管理：conda。"),
        ("我偏好小批量训练。", "用户训练偏好：小批量。"),
        ("学习率我一般用 1e-3。", "用户常用学习率：1e-3。"),
        ("你记得我用的编辑器吗？", None),
        ("我学的数学课？", None),
        ("牛顿第二定律公式？", None),
        ("欧拉公式？", None),
        ("勾股定理？", None),
        ("SVD 是什么？", None),
        ("我记笔记的方式？", None),
        ("我的学习率习惯？", None),
    ]
    steps.extend(extra)
    # 再补到至少 TOTAL_ROUNDS 轮
    while len(steps) < TOTAL_ROUNDS:
        steps.append((f"补充记忆第 {len(steps)+1} 条。", f"用户说：补充记忆第 {len(steps)+1} 条。"))
    return steps[:TOTAL_ROUNDS]


def make_config():
    return PNMSConfig(
        cold_start_n0=COLD_START_N0,
        cold_start_n1=COLD_START_N1,
        max_slots_per_user=200,
        retrieval_top_k=8,
        max_context_slots=10,
        graph_enabled=True,
        graph_max_neighbors_per_slot=4,
        graph_max_expanded_total=18,
        concept_enabled=True,
        concept_min_cluster_size=CONCEPT_MIN_CLUSTER,
        concept_formation_interval_rounds=CONCEPT_FORMATION_INTERVAL,
        concept_checkpoint_dir=str(CONCEPT_CKPT_DIR),
        write_new_slot_threshold=0.98,
        merge_similarity_threshold=0.99,
        decay_every_n_rounds=15,
    )


def phase1_train_and_save():
    """阶段一：多轮对话，形成概念并保存。"""
    print("=== 阶段一：大样本训练与保存 ===\n")
    config = make_config()
    device = torch.device("cpu")
    encoder = SimpleQueryEncoder(config.embed_dim or 64, vocab_size=10000)
    pnms = PNMS(config=config, user_id="user_1", encoder=encoder, device=device)
    if hasattr(pnms.encoder, "eval"):
        pnms.encoder.eval()

    steps = build_large_steps()
    for i, (query, content) in enumerate(steps):
        pnms.handle_query(
            query,
            llm=mock_llm,
            content_to_remember=content,
            system_prompt="你是个人助手。",
        )
        if (i + 1) % 20 == 0:
            print(f"  已执行 {i+1}/{len(steps)} 轮，槽数={pnms.store.num_slots}，概念模块数={len(pnms.concept_manager._modules)}")

    num_edges = len(pnms.graph._edges)
    print(f"\n阶段一结束：槽数={pnms.store.num_slots}，概念模块数={len(pnms.concept_manager._modules)}，图边数={num_edges}")
    if pnms.concept_manager._modules:
        pnms.save_concept_modules()
        graph_db = CONCEPT_CKPT_DIR / "graph.db"
        print(f"已保存概念模型与图到: {CONCEPT_CKPT_DIR}（含 graph.db: {graph_db.exists()}）")
    else:
        print("未形成概念模块（槽数或间隔未满足条件），仍执行阶段二以验证加载路径。")
    return pnms.store.num_slots, len(pnms.concept_manager._modules), num_edges


def phase2_load_and_verify():
    """阶段二：新建 PNMS 加载概念模型与图，复现槽后推理验证。"""
    print("\n=== 阶段二：加载模型并推理验证 ===\n")
    config = make_config()
    device = torch.device("cpu")
    encoder = SimpleQueryEncoder(config.embed_dim or 64, vocab_size=10000)
    pnms2 = PNMS(config=config, user_id="user_1", encoder=encoder, device=device)
    if hasattr(pnms2.encoder, "eval"):
        pnms2.encoder.eval()

    loaded_concepts = len(pnms2.concept_manager._modules)
    loaded_edges = len(pnms2.graph._edges)
    print(f"启动后加载: 概念模块数={loaded_concepts}, 图边数={loaded_edges}")

    steps = build_large_steps()
    for i, (query, content) in enumerate(steps):
        pnms2.handle_query(
            query,
            llm=mock_llm,
            content_to_remember=content,
            system_prompt="你是个人助手。",
        )
    edges_after_replay = len(pnms2.graph._edges)
    print(f"复现 {len(steps)} 轮后槽数: {pnms2.store.num_slots}，图边数: {edges_after_replay}（加载时 {loaded_edges}，复现中新增共现会累加）")

    system_prompt = "你是个人助手。请严格依据上文的「记忆」回答。"
    verification_queries = [
        "你记得我喜欢用什么语言吗？",
        "二次方程求根公式？",
        "我用的开发环境？",
        "牛顿第二定律公式？",
        "我记笔记的方式？",
    ]

    print("\n--- 推理验证：同一 query 下「不启用概念」vs「启用概念」的上下文对比 ---\n")
    all_ok = True
    for idx, q in enumerate(verification_queries):
        ctx_no, n_no = pnms2.get_context_for_query(q, system_prompt=system_prompt, use_concept=False)
        ctx_yes, n_yes = pnms2.get_context_for_query(q, system_prompt=system_prompt, use_concept=True)
        print(f"Query: {q}")
        print(f"  不启用概念: 槽数={n_no}, 字符数={len(ctx_no)}")
        print(f"  启用概念:   槽数={n_yes}, 字符数={len(ctx_yes)}")
        if idx == 0 and n_yes > 0:
            snippet = ctx_yes.replace(system_prompt, "").strip()[:200]
            print(f"  上下文摘要: {snippet}...")
        if loaded_concepts > 0 and n_yes > 0:
            if ctx_no != ctx_yes:
                print("  结论: 概念模块参与重排，上下文与未启用时不同。")
            else:
                print("  结论: 上下文相同（检索到的槽中属于概念簇的未改变顺序）。")
        if n_yes == 0 and n_no == 0:
            print("  注意: 未检索到槽，请确认复现轮次足够。")
            all_ok = False
        print()
    return loaded_concepts, loaded_edges, all_ok


def main():
    CONCEPT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    num_slots, num_concepts, num_edges_p1 = phase1_train_and_save()
    loaded_concepts, loaded_edges, verify_ok = phase2_load_and_verify()

    print("=== 自测总结 ===")
    print(f"阶段一: 槽数={num_slots}, 概念数={num_concepts}, 图边数={num_edges_p1}, 已保存到 {CONCEPT_CKPT_DIR}")
    print(f"阶段二: 加载概念数={loaded_concepts}, 加载图边数={loaded_edges}, 推理验证={'通过' if verify_ok else '有告警'}")
    ok_concept = num_concepts == 0 or loaded_concepts == num_concepts
    ok_graph = num_edges_p1 == loaded_edges
    if ok_concept and ok_graph and verify_ok:
        print("自测结果: 全部通过（概念保存/加载、图保存/加载、推理验证）。")
    else:
        if not ok_concept:
            print("自测失败: 概念数量与加载不一致。")
        if not ok_graph:
            print("自测失败: 阶段一有图边但阶段二加载后图边为 0，或反之。")
        if not verify_ok:
            print("自测告警: 推理验证存在槽数为 0 等。")


if __name__ == "__main__":
    main()
