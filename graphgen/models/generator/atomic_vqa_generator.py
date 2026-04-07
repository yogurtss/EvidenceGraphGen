from typing import Any

from graphgen.bases import BaseGenerator

from .atomic_generator import AtomicGenerator


class AtomicVQAGenerator(AtomicGenerator):
    """Atomic generator with VQA-compatible image output."""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> list[dict]:
        qa_pairs = await BaseGenerator.generate(self, batch)
        if not qa_pairs:
            return []

        img_path = self.extract_visual_asset_path(batch)
        if img_path:
            for qa in qa_pairs:
                qa["img_path"] = img_path
        return qa_pairs

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return BaseGenerator.format_vqa_generation_results(result, output_data_format)
