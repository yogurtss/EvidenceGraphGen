import json

from .aggregated_generator import AggregatedGenerator


class AggregatedVQAGenerator(AggregatedGenerator):
    """Aggregated two-step generator with VQA-compatible image output."""

    @staticmethod
    def _extract_img_path(nodes: list[tuple[str, dict]]) -> str:
        for _, node_data in nodes:
            if "metadata" not in node_data or not node_data["metadata"]:
                continue
            try:
                raw_metadata = json.loads(node_data["metadata"])
            except (json.JSONDecodeError, TypeError):
                continue
            metadata = (
                raw_metadata.get("metadata", {})
                if isinstance(raw_metadata.get("metadata"), dict)
                else raw_metadata
            )
            img_path = metadata.get("img_path") or metadata.get("path", "")
            if img_path:
                return img_path
        return ""

    async def generate(self, batch):
        qa_pairs = await super().generate(batch)
        if not qa_pairs:
            return []
        nodes, _ = batch
        img_path = self._extract_img_path(nodes)
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
