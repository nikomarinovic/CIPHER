from brain.reasoning import ReasoningEngine
from brain.internet import InternetEngine
from brain.synthesizer import AnswerSynthesizer


class CIPHERBrain:

    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.internet = InternetEngine()
        self.synthesizer = AnswerSynthesizer()

    def think(self, user_input: str):

        thought = self.reasoning.analyze(
            user_input,
            []
        )

        if thought.confidence >= 1.0:
            return thought

        results = self.internet.search(
            user_input,
            limit=5
        )

        if not results:
            thought.answer = (
                "I couldn't find useful information about that."
            )

            thought.reasoning.append(
                "Internet search returned no useful results."
            )

            return thought

        synthesized = self.synthesizer.synthesize(
            user_input,
            results
        )

        thought.answer = synthesized["answer"]
        thought.sources = synthesized["sources"]
        thought.confidence = 0.7

        thought.reasoning.append(
            f"Internet search returned {len(results)} results."
        )

        thought.reasoning.append(
            "Answer synthesized from search results."
        )

        return thought