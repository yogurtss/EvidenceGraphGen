from graphgen.bases import BaseGenerator
from .aggregated_generator import AggregatedGenerator


class AggregatedVQAGenerator(AggregatedGenerator):
    """Aggregated two-step generator with VQA-compatible image output."""

    async def generate(self, batch):
        rephrasing_prompt = self.build_prompt(
            batch,
            include_source_chunks_in_prompt=self.include_source_chunks_in_prompt,
            source_chunk_context_builder=self.source_chunk_context_builder,
        )
        img_path = self.extract_visual_asset_path(batch)
        response = await self.llm_client.generate_answer(
            rephrasing_prompt, image_path=img_path or None
        )
        context = self.parse_rephrased_text(response)
        if not context:
            return []
        question_generation_prompt = self._build_prompt_for_question_generation(context)
        response = await self.llm_client.generate_answer(
            question_generation_prompt, image_path=img_path or None
        )
        question = self.parse_response(response)["question"]
        if not question:
            return []

        qa_pair = {
            "question": question,
            "answer": context,
        }
        if img_path:
            qa_pair["img_path"] = img_path
        return [qa_pair]

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        return BaseGenerator.format_vqa_generation_results(result, output_data_format)
