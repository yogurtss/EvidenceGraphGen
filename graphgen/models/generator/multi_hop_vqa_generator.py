from typing import Any

from graphgen.bases import BaseGenerator

from .multi_hop_generator import MultiHopGenerator


class MultiHopVQAGenerator(MultiHopGenerator):
    """Multi-hop generator with VQA-compatible image output."""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> list[dict]:
        prompt = self.build_prompt(
            batch,
            include_source_chunks_in_prompt=self.include_source_chunks_in_prompt,
            source_chunk_context_builder=self.source_chunk_context_builder,
        )
        img_path = self.extract_visual_asset_path(batch)
        response = await self.llm_client.generate_answer(
            prompt,
            image_path=img_path or None,
        )
        qa_pairs = self.parse_response(response)
        if not qa_pairs:
            return []

        if img_path:
            for qa in qa_pairs:
                qa["img_path"] = img_path
        return qa_pairs

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return BaseGenerator.format_vqa_generation_results(result, output_data_format)
