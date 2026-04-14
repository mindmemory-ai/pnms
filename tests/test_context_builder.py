import unittest

from pnms import ContextBuilder, MemorySlot
import torch


class ContextBuilderTests(unittest.TestCase):
    def _make_slot(self, content: str) -> MemorySlot:
        # 使用一个简单的恒定向量作为 key，ContextBuilder 不会依赖 key 的具体值
        key = torch.ones(4)
        return MemorySlot.create(
            user_id="u1",
            key=key,
            content=content,
            weight=1.0,
        )

    def test_max_context_slots_limit(self):
        builder = ContextBuilder(max_context_slots=2, max_context_tokens=None)
        slots = [self._make_slot(f"c{i}") for i in range(3)]
        text = builder.build(slots, system_prompt=None)
        # 只应包含前两个槽
        self.assertIn("记忆：c0", text)
        self.assertIn("记忆：c1", text)
        self.assertNotIn("记忆：c2", text)

    def test_token_budget_truncation(self):
        # 自定义 token_counter，使每条记忆占用 1 token，预算为 2，只能留前两条
        builder = ContextBuilder(
            max_context_slots=10,
            max_context_tokens=2,
            token_counter=lambda s: 1,
        )
        slots = [self._make_slot(f"c{i}") for i in range(5)]
        text = builder.build(slots, system_prompt=None)
        self.assertIn("记忆：c0", text)
        self.assertIn("记忆：c1", text)
        self.assertNotIn("记忆：c2", text)

    def test_context_supplement_and_system_prompt(self):
        builder = ContextBuilder(max_context_slots=1)
        slots = [self._make_slot("slot")]
        text = builder.build(
            slots,
            state_summary="user summary",
            system_prompt="sys",
            context_supplement="supp",
        )
        self.assertIn("sys", text)
        self.assertIn("[用户摘要] user summary", text)
        self.assertIn("[补充上下文]\nsupp", text)
        self.assertIn("记忆：slot", text)


if __name__ == "__main__":
    unittest.main()

