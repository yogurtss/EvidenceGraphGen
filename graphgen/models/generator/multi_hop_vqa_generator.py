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
        qa_pairs = await BaseGenerator.generate(self, batch)
        if not qa_pairs:
            return []

        img_path = self.extract_image_path(batch)
        for qa in qa_pairs:
            qa["img_path"] = img_path
        return qa_pairs

    @staticmethod
    def format_generation_results(result: dict, output_data_format: str) -> dict:
        question = result.get("question", "")
        answer = result.get("answer", "")
        img_path = result.get("img_path", "")
        if output_data_format == "Alpaca":
            output = {
                "instruction": question,
                "input": "",
                "output": answer,
            }
            if img_path:
                output["image"] = img_path
            return output
        if output_data_format == "Sharegpt":
            user_value = [{"text": question}]
            if img_path:
                user_value[0]["image"] = img_path
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": user_value,
                    },
                    {"from": "gpt", "value": [{"text": answer}]},
                ]
            }
        if output_data_format == "ChatML":
            user_content = [{"text": question}]
            if img_path:
                user_content[0]["image"] = img_path
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]
            }
        raise ValueError(f"Unknown output data format: {output_data_format}")
