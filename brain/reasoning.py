from dataclasses import dataclass, field
import re
import ast
import operator

@dataclass
class Thought:
    input_text: str
    intent: str
    topic: str
    answer: str
    confidence: float
    reasoning: list[str]
    sources: list[dict] = field(default_factory=list)
    preformatted: bool = False


class ReasoningEngine:

    def analyze(self, user_input: str, context: list[str]) -> Thought:
        text = user_input.strip()

        reasoning = [
            "Input received.",
            f"Input length: {len(text)} characters."
        ]

        expression = self.extract_expression(text)

        if expression:
            reasoning.append(
                f"Mathematical expression detected: {expression}"
            )

            answer = self.solve(expression)

            if answer is not None:
                reasoning.append(
                    "Expression evaluated successfully."
                )

                return Thought(
                    input_text=text,
                    intent="calculation",
                    topic="mathematics",
                    answer=str(answer),
                    confidence=1.0,
                    reasoning=reasoning
                )

        reasoning.append(
            "No local answer available."
        )

        return Thought(
            input_text=text,
            intent="unknown",
            topic="unknown",
            answer=None,
            confidence=0.0,
            reasoning=reasoning
        )

    def extract_expression(self, text: str):
        matches = re.findall(
            r'\d+(?:\.\d+)?(?:\s*[+\-*/%]\s*\d+(?:\.\d+)?)+',
            text
        )

        if not matches:
            return None

        return matches[-1]

    def solve(self, expression: str):
        try:
            tree = ast.parse(
                expression,
                mode="eval"
            )

            return self.evaluate_node(tree.body)

        except Exception:
            return None

    def evaluate_node(self, node):

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
        }

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value

        if isinstance(node, ast.BinOp):
            left = self.evaluate_node(node.left)
            right = self.evaluate_node(node.right)

            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator")

            return operation(left, right)

        raise ValueError("Unsupported expression")