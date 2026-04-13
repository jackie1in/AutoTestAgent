"""Form Combination Tester - 表单字段组合测试"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from graph_agent.models import Transition

logger = logging.getLogger(__name__)


@dataclass
class FormField:
    """表单字段定义"""
    index: int
    name: str
    field_type: str  # input, select, textarea, checkbox, etc.
    is_required: bool = False
    validations: list[str] = field(default_factory=list)
    test_values: list[Any] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.test_values:
            self.test_values = self._generate_default_test_values()
    
    def _generate_default_test_values(self) -> list[Any]:
        """生成默认测试值"""
        if self.field_type in ('text', 'input'):
            return ['', 'test', 'test@example.com', 'a' * 100]
        elif self.field_type == 'number':
            return [0, -1, 999999, 3.14]
        elif self.field_type == 'select':
            return [0, 1, -1]
        elif self.field_type == 'checkbox':
            return [True, False]
        elif self.field_type == 'date':
            return ['2024-01-01', 'invalid', '']
        else:
            return ['test', '']


class FormCombinationGenerator:
    """表单组合生成器"""
    
    def __init__(self, fields: list[FormField]):
        self._fields = fields
    
    def generate_combinations(self, strategy: str = "pairwise") -> list[dict]:
        """
        生成测试组合
        
        策略:
        - all: 所有组合
        - pairwise: 两两组合（推荐）
        - required_only: 只测必填字段
        """
        combinations = []
        
        if strategy == "all" and len(self._fields) <= 5:
            # 笛卡尔积
            field_values = [(f.index, f.test_values) for f in self._fields]
            for values in itertools.product(*[v for _, v in field_values]):
                combo = {field_values[i][0]: v for i, v in enumerate(values)}
                combinations.append(combo)
        
        elif strategy == "pairwise":
            # 基础组合
            base = {f.index: f.test_values[0] if f.test_values else '' 
                   for f in self._fields}
            combinations.append(base.copy())
            
            # 每个字段变化
            for field in self._fields:
                for val in field.test_values[1:3]:  # 限制数量
                    combo = {**base, field.index: val}
                    combinations.append(combo)
        
        elif strategy == "required_only":
            required = [f for f in self._fields if f.is_required]
            combo = {f.index: f.test_values[0] if f.test_values else '' 
                    for f in required}
            combinations.append(combo)
        
        return combinations


class FormTester:
    """表单测试器"""
    
    def __init__(self, session, controller):
        self._session = session
        self._controller = controller
    
    async def test_form_combinations(
        self,
        form_fields: list[FormField],
        submit_button_index: int,
        strategy: str = "pairwise",
    ) -> list[Transition]:
        """测试表单的所有组合"""
        generator = FormCombinationGenerator(form_fields)
        combinations = generator.generate_combinations(strategy)
        
        logger.info(f"Testing {len(combinations)} form combinations")
        
        transitions = []
        
        for i, combo in enumerate(combinations):
            try:
                # 填充表单
                for field_idx, value in combo.items():
                    await self._fill_field(field_idx, value)
                
                # 提交
                await self._controller.click_element(submit_button_index)
                
                # 记录结果
                # ...
                
            except Exception as e:
                logger.error(f"Combination {i} failed: {e}")
        
        return transitions
    
    async def _fill_field(self, field_idx: int, value: Any):
        """填充单个字段"""
        await self._controller.input_text(field_idx, str(value))
