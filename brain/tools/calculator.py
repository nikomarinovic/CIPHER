import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def calculate(expression: str):
    """
    Safely evaluate a mathematical expression.
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return result

    except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
        return None


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Invalid value")

    if isinstance(node, ast.BinOp):
        operator_function = OPERATORS.get(type(node.op))

        if operator_function is None:
            raise ValueError("Unsupported operator")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        return operator_function(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)

        if isinstance(node.op, ast.USub):
            return -operand

        if isinstance(node.op, ast.UAdd):
            return operand

    raise ValueError("Unsupported expression")