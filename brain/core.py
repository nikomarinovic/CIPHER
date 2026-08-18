from brain.reasoning import ReasoningEngine, Thought
from brain.internet import InternetEngine
from brain.synthesizer import AnswerSynthesizer
from brain.router import Router
from brain.tools.geo import GeoRouter, draw_route_map


class CIPHERBrain:

    def __init__(self):
        self.reasoning = ReasoningEngine()
        self.internet = InternetEngine()
        self.synthesizer = AnswerSynthesizer()
        self.router = Router()
        self.geo = GeoRouter()

    def think(self, user_input: str):

        route_thought = self._try_route(user_input)

        if route_thought is not None:
            return route_thought

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

    def _try_route(self, user_input: str):
        """Check whether the input is a 'from A to B' style route query
        (e.g. 'koliko je od Zagreba do Splita'). If so, geocode both
        places and fetch route alternatives instead of falling through
        to web search — routing isn't something free-text search answers
        well, but a dedicated routing API handles it directly."""
        routed = self.router.route(user_input)

        if routed["type"] != "route_query":
            return None

        origin_name = routed["origin"]
        destination_name = routed["destination"]

        reasoning = [
            f"Route query detected: '{origin_name}' -> '{destination_name}'."
        ]

        origin = self.geo.geocode(origin_name)
        destination = self.geo.geocode(destination_name)

        if not origin or not destination:
            missing = origin_name if not origin else destination_name

            reasoning.append(f"Could not geocode '{missing}'.")

            return Thought(
                input_text=user_input,
                intent="route",
                topic="geography",
                answer=(
                    f"Nisam uspio pronaci lokaciju '{missing}'. "
                    "Provjeri je li naziv mjesta točno napisan."
                ),
                confidence=1.0,
                reasoning=reasoning,
            )

        routes = self.geo.get_routes(origin, destination)

        if not routes:
            reasoning.append("Routing service returned no routes.")

            return Thought(
                input_text=user_input,
                intent="route",
                topic="geography",
                answer=(
                    "Pronašao sam obje lokacije, ali trenutno ne mogu "
                    "izračunati rutu — usluga za rute nije dostupna ili "
                    "mjesta nisu povezana cestom."
                ),
                confidence=1.0,
                reasoning=reasoning,
            )

        reasoning.append(f"Found {len(routes)} route option(s).")

        # The map plots the fastest route's actual road geometry — the
        # other alternatives (if any) are mentioned as text below it,
        # since drawing 2-3 overlapping paths on a small text canvas
        # gets unreadable fast.
        fastest = routes[0]
        answer = draw_route_map(origin, destination, fastest)

        if len(routes) > 1:
            extra_lines = ["", "Ostale opcije:"]

            for index, alternative in enumerate(routes[1:], start=1):
                duration_min = alternative["duration_min"]
                hours, minutes = divmod(int(round(duration_min)), 60)
                duration_text = f"{hours}h {minutes}min" if hours else f"{minutes} min"
                extra_lines.append(
                    f"  {index}. {alternative['distance_km']:.1f} km — {duration_text}"
                )

            answer += "\n" + "\n".join(extra_lines)

        return Thought(
            input_text=user_input,
            intent="route",
            topic="geography",
            answer=answer,
            confidence=1.0,
            reasoning=reasoning,
            sources=[
                {"title": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org"},
                {"title": "OSRM Routing", "url": "https://project-osrm.org"},
            ],
            preformatted=True,
        )