"""数学工具。"""

import ast
import operator

from .errors import ToolExecutionError

SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_eval(expr: str) -> float:
    """只允许数字、括号、四则运算和安全负号表达式求值。"""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return operator.neg(_eval(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPERATORS:
            left = _eval(node.left)
            right = _eval(node.right)
            return SAFE_BINARY_OPERATORS[type(node.op)](left, right)

        raise ValueError(f"不支持的表达式：{ast.dump(node)}")

    return _eval(tree)


async def calculate(expression: str) -> str:
    """计算安全数学表达式。"""
    expression = expression.strip()

    if not expression:
        raise ToolExecutionError("无法识别数学表达式")

    try:
        result = safe_eval(expression)
    except SyntaxError as e:
        raise ToolExecutionError(f"表达式语法错误：{e}") from e
    except ZeroDivisionError as e:
        raise ToolExecutionError(f"表达式有误：{e}") from e
    except ValueError as e:
        raise ToolExecutionError(f"错误：{e}") from e
    except Exception as e:
        raise ToolExecutionError(f"未知错误：{type(e).__name__}: {e}") from e

    return str(result)