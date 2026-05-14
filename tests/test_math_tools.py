"""math_tools.py 测试模块"""

import asyncio

import pytest

from src.tools.math_tools import calculate, safe_eval
from src.tools.errors import ToolExecutionError


class TestSafeEval:
    """safe_eval() 函数直接测试"""

    class TestNormal:
        """正常四则运算测试"""

        def test_addition(self):
            assert safe_eval("2 + 3") == 5

        def test_subtraction(self):
            assert safe_eval("10 - 4") == 6

        def test_multiplication(self):
            assert safe_eval("6 * 7") == 42

        def test_division(self):
            assert safe_eval("20 / 4") == 5.0

        def test_complex_expression(self):
            assert safe_eval("(2 + 3) * 4 - 10 / 2") == 15.0

    class TestWithNegative:
        """含负数运算测试"""

        def test_negative_number(self):
            assert safe_eval("-5 + 3") == -2

        def test_subtract_to_negative(self):
            assert safe_eval("3 - 10") == -7

        def test_multiply_negatives(self):
            assert safe_eval("-3 * -4") == 12

        def test_mixed_negative(self):
            assert safe_eval("(-2 + 5) * (-3)") == -9

    class TestWithParentheses:
        """含括号运算测试"""

        def test_simple_parentheses(self):
            assert safe_eval("(2 + 3) * 4") == 20

        def test_nested_parentheses(self):
            assert safe_eval("((1 + 2) * (3 + 4))") == 21

        def test_deeply_nested_parentheses(self):
            assert safe_eval("(((1 + 2) + 3) + 4)") == 10

    class TestDivisionByZero:
        """除以零异常测试"""

        def test_simple_division_by_zero(self):
            with pytest.raises(ZeroDivisionError):
                safe_eval("10 / 0")

        def test_expression_division_by_zero(self):
            with pytest.raises(ZeroDivisionError):
                safe_eval("5 / (2 - 2)")

    class TestInvalidExpression:
        """非法表达式异常测试"""

        def test_syntax_error(self):
            with pytest.raises(Exception):
                safe_eval("2 + * 3")

        def test_unsupported_operator(self):
            with pytest.raises(ValueError):
                safe_eval("2 ** 3")

        def test_variable_in_expression(self):
            with pytest.raises(ValueError):
                safe_eval("2 + x")

        def test_builtin_call(self):
            """内置函数调用应被拒绝"""
            with pytest.raises(ValueError):
                safe_eval("abs(-1)")


class TestCalculateNormal:
    """正常四则运算测试"""

    def test_addition(self):
        result = asyncio.run(calculate("2 + 3"))
        assert result == "5"

    def test_subtraction(self):
        result = asyncio.run(calculate("10 - 4"))
        assert result == "6"

    def test_multiplication(self):
        result = asyncio.run(calculate("6 * 7"))
        assert result == "42"

    def test_division(self):
        result = asyncio.run(calculate("20 / 4"))
        assert result == "5.0"

    def test_complex_expression(self):
        result = asyncio.run(calculate("(2 + 3) * 4 - 10 / 2"))
        assert result == "15.0"

    def test_nested_parentheses(self):
        result = asyncio.run(calculate("((2 + 3) * (4 - 1)) / 3"))
        assert result == "5.0"


class TestCalculateWithNegative:
    """含负数运算测试"""

    def test_negative_number(self):
        result = asyncio.run(calculate("-5 + 3"))
        assert result == "-2"

    def test_subtract_to_negative(self):
        result = asyncio.run(calculate("3 - 10"))
        assert result == "-7"

    def test_multiply_negatives(self):
        result = asyncio.run(calculate("-3 * -4"))
        assert result == "12"

    def test_mixed_negative(self):
        result = asyncio.run(calculate("(-2 + 5) * (-3)"))
        assert result == "-9"


class TestCalculateWithParentheses:
    """含括号运算测试"""

    def test_simple_parentheses(self):
        result = asyncio.run(calculate("(2 + 3) * 4"))
        assert result == "20"

    def test_nested_parentheses(self):
        result = asyncio.run(calculate("((1 + 2) * (3 + 4))"))
        assert result == "21"

    def test_deeply_nested_parentheses(self):
        result = asyncio.run(calculate("(((1 + 2) + 3) + 4)"))
        assert result == "10"

    def test_parentheses_with_negative(self):
        result = asyncio.run(calculate("(-5 + 3) * (2 - 4)"))
        assert result == "4"


class TestCalculateDivisionByZero:
    """除以零异常测试"""

    def test_simple_division_by_zero(self):
        with pytest.raises(ToolExecutionError, match="表达式有误"):
            asyncio.run(calculate("10 / 0"))

    def test_expression_division_by_zero(self):
        with pytest.raises(ToolExecutionError, match="表达式有误"):
            asyncio.run(calculate("5 / (2 - 2)"))


class TestCalculateInvalidExpression:
    """非法表达式异常测试"""

    def test_syntax_error(self):
        with pytest.raises(ToolExecutionError, match="语法错误"):
            asyncio.run(calculate("2 + * 3"))

    def test_unsupported_operator(self):
        with pytest.raises(ToolExecutionError, match="不支持"):
            asyncio.run(calculate("2 ** 3"))

    def test_variable_in_expression(self):
        with pytest.raises(ToolExecutionError, match="不支持"):
            asyncio.run(calculate("2 + x"))


class TestCalculateDangerousExpression:
    """危险性高的表达式异常测试"""

    def test_attribute_access(self):
        with pytest.raises(ToolExecutionError):
            asyncio.run(calculate("(1).__class__"))

    def test_abs_builtin(self):
        """内置函数调用应被拒绝"""
        with pytest.raises(ToolExecutionError):
            asyncio.run(calculate("abs(-1)"))


class TestCalculateEmptyExpression:
    """空字符表达式异常测试"""

    def test_empty_string(self):
        with pytest.raises(ToolExecutionError, match="无法识别数学表达式"):
            asyncio.run(calculate(""))

    def test_whitespace_only(self):
        with pytest.raises(ToolExecutionError, match="无法识别数学表达式"):
            asyncio.run(calculate("   "))
