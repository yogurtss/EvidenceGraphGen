import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates import (
    MULTI_HOP_GENERATION_PROMPT,
    MULTI_HOP_GENERATION_PROMPT_WITH_SOURCE_CONTEXT,
)
from graphgen.utils import detect_main_language, logger

from .context_utils import build_grounded_context
from .source_context import SourceChunkContextBuilder


class MultiHopGenerator(BaseGenerator):
    def __init__(
        self,
        llm_client,
        *,
        include_source_chunks_in_prompt: bool = False,
        source_chunk_storages: list[Any] | None = None,
        source_chunks_per_entity: int = 3,
    ):
        super().__init__(llm_client)
        self.include_source_chunks_in_prompt = bool(include_source_chunks_in_prompt)
        self.source_chunk_context_builder = SourceChunkContextBuilder(
            source_chunk_storages,
            chunks_per_entity=source_chunks_per_entity,
        )

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]],
        *,
        include_source_chunks_in_prompt: bool = False,
        source_chunk_context_builder: SourceChunkContextBuilder | None = None,
    ) -> str:
        entities_str, relationships_str = build_grounded_context(
            batch,
            include_visual_metadata=True,
        )
        language = detect_main_language(entities_str + relationships_str)
        source_chunks = (
            source_chunk_context_builder.build(batch)
            if include_source_chunks_in_prompt and source_chunk_context_builder
            else ""
        )
        if source_chunks:
            prompt = MULTI_HOP_GENERATION_PROMPT_WITH_SOURCE_CONTEXT[language].format(
                entities=entities_str,
                relationships=relationships_str,
                source_chunks=source_chunks,
            )
        else:
            prompt = MULTI_HOP_GENERATION_PROMPT[language].format(
                entities=entities_str, relationships=relationships_str
            )
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()
        else:
            logger.warning("Failed to parse response: %s", response)
            return []

        question = question.strip('"').strip("'")
        answer = answer.strip('"').strip("'")
        logger.debug("Question: %s", question)
        logger.debug("Answer: %s", answer)
        return [{"question": question, "answer": answer}]

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> list[dict]:
        """Text-only multi-hop generation without image injection."""
        prompt = self.build_prompt(
            batch,
            include_source_chunks_in_prompt=self.include_source_chunks_in_prompt,
            source_chunk_context_builder=self.source_chunk_context_builder,
        )
        response = await self.llm_client.generate_answer(prompt)
        return self.parse_response(response)
