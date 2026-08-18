import re

from brain.tools.calculator import calculate


class Router:

    # Ordered from most specific to most generic, so a phrase like
    # "koliko je od Zagreba do Splita" matches the Croatian-specific
    # pattern before falling through to the generic "od X do Y" one.
    ROUTE_PATTERNS = [
        re.compile(r"(?:koliko je|kolika je udaljenost|udaljenost)\s+(?:od\s+)?(.+?)\s+do\s+(.+)$", re.IGNORECASE),
        re.compile(r"(?:ruta|rutu|put|najbrza ruta|najbrzi put)\s+od\s+(.+?)\s+do\s+(.+)$", re.IGNORECASE),
        re.compile(r"^od\s+(.+?)\s+do\s+(.+)$", re.IGNORECASE),
        re.compile(r"(?:route|directions|distance)\s+from\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE),
        re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE),
        re.compile(r"von\s+(.+?)\s+nach\s+(.+)$", re.IGNORECASE),
    ]

    def __init__(self):
        self.calculator = calculate

    def route(self, user_input: str):
        route_match = self._extract_route(user_input)

        if route_match is not None:
            origin, destination = route_match

            return {
                "type": "route_query",
                "origin": origin,
                "destination": destination,
            }

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

    def _extract_route(self, text: str):
        cleaned = text.strip().rstrip("?.!").strip()

        if not cleaned:
            return None

        for pattern in self.ROUTE_PATTERNS:
            match = pattern.search(cleaned)

            if not match:
                continue

            origin = match.group(1).strip(" ,.")
            destination = match.group(2).strip(" ,.")

            if origin and destination:
                return origin, destination

        return None

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