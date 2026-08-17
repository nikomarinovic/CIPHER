import math

from brain.nlp_utils import tokenize, split_sentences, phrase_proximity_bonus


class AnswerSynthesizer:

    MIN_SENTENCE_LENGTH = 30
    MAX_SENTENCE_LENGTH = 400
    MAX_SENTENCES = 6
    DUPLICATE_THRESHOLD = 0.6
    DOMINANCE_RATIO = 1.6

    def synthesize(self, query: str, results: list[dict]) -> dict:
        if not results:
            return {
                "answer": "I couldn't find any useful information about that.",
                "sources": []
            }

        query_tokens = tokenize(query)

        results = sorted(results, key=lambda item: item.get("relevance", 0), reverse=True)

        if len(results) > 1 and results[0].get("relevance", 0) > 0:
            if results[0]["relevance"] >= results[1].get("relevance", 0) * self.DOMINANCE_RATIO:
                results = results[:1]

        sentence_pool = self._build_sentence_pool(results)

        scored = self._score_sentences(sentence_pool, query_tokens)

        if not scored:
            return {
                "answer": "I found some sources but couldn't extract a clear answer from them.",
                "sources": self._build_sources(results, {0})
            }

        selected = self._select_sentences(scored)

        used_source_indices = {sentence["source_index"] for sentence in selected}

        selected.sort(key=lambda sentence: (sentence["source_index"], sentence["position"]))

        answer = self._build_paragraph(selected)

        return {
            "answer": answer,
            "sources": self._build_sources(results, used_source_indices)
        }

    def _build_sentence_pool(self, results: list[dict]) -> list[dict]:
        pool = []

        for source_index, result in enumerate(results):
            text = result.get("content") or result.get("snippet", "")
            sentences = split_sentences(text)
            source_weight = 1.0 / (1.0 + source_index * 0.4)

            for position, sentence in enumerate(sentences):
                length = len(sentence)

                if length < self.MIN_SENTENCE_LENGTH or length > self.MAX_SENTENCE_LENGTH:
                    continue

                pool.append({
                    "text": sentence,
                    "tokens": set(tokenize(sentence)),
                    "source_index": source_index,
                    "position": position,
                    "source_weight": source_weight
                })

        return pool

    def _score_sentences(self, sentence_pool: list[dict], query_tokens: list[str]) -> list[dict]:
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        total = max(len(sentence_pool), 1)
        document_frequency = {}

        for sentence in sentence_pool:
            for token in sentence["tokens"]:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        scored = []

        for sentence in sentence_pool:
            overlap = sentence["tokens"] & query_set

            if not overlap:
                continue

            score = 0.0

            for token in overlap:
                term_frequency = sentence["text"].lower().count(token)
                inverse_document_frequency = math.log(
                    (total + 1) / (document_frequency.get(token, 1) + 1)
                ) + 1
                score += term_frequency * inverse_document_frequency

            score *= sentence["source_weight"]
            score *= (1 + 0.15 * len(overlap))
            score *= 1.0 / (1.0 + sentence["position"] * 0.05)
            score *= phrase_proximity_bonus(query_tokens, sentence["text"])

            sentence["score"] = score
            scored.append(sentence)

        scored.sort(key=lambda item: item["score"], reverse=True)

        return scored

    def _select_sentences(self, scored_sentences: list[dict]) -> list[dict]:
        selected = []

        for sentence in scored_sentences:
            if len(selected) >= self.MAX_SENTENCES:
                break

            is_duplicate = False

            for chosen in selected:
                union = sentence["tokens"] | chosen["tokens"]

                if not union:
                    continue

                similarity = len(sentence["tokens"] & chosen["tokens"]) / len(union)

                if similarity >= self.DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(sentence)

        return selected

    def _build_paragraph(self, sentences: list[dict]) -> str:
        # Always return one continuous paragraph — no mid-answer split.
        texts = [sentence["text"] for sentence in sentences]
        return " ".join(texts)

    def _build_sources(self, results: list[dict], used_indices: set) -> list[dict]:
        sources = []
        seen_urls = set()

        for index in sorted(used_indices):
            if index >= len(results):
                continue

            result = results[index]
            title = result.get("title", "").strip()
            url = result.get("url", "").strip()

            if title and url and url not in seen_urls:
                sources.append({"title": title, "url": url})
                seen_urls.add(url)

        return sources