from brain.tools.calculator import calculate


class Router:

    def __init__(self):
        self.calculator = calculate

    def route(self, user_input: str):
        expression = self._extract_math_expression(user_input)

        if expression is not None:
            result = self.calculator(expression)

            if result is not None:
                return {
                    "type": "calculation",
                    "result": result,
                }

        return {
            "type": "unknown",
            "result": None,
        }

    def _extract_math_expression(self, text: str):
        expression = "".join(
            character
            for character in text
            if character.isdigit()
            or character in "+-*/().%"
            or character.isspace()
        )

        expression = expression.strip()

        if not expression:
            return None

        if not any(
            operator in expression
            for operator in "+-*/%"
        ):
            return None

        return expression