from abc import ABC, abstractmethod
import json
from typing import Any

from graphgen.bases.base_llm_wrapper import BaseLLMWrapper


class BaseGenerator(ABC):
    """
    Generate QAs based on given prompts.
    """

    def __init__(self, llm_client: BaseLLMWrapper):
        self.llm_client = llm_client

    @staticmethod
    @abstractmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        """Build prompt for LLM based on the given batch"""

    @staticmethod
    @abstractmethod
    def parse_response(response: str) -> list[dict]:
        """Parse the LLM response and return the generated QAs"""

    @staticmethod
    def load_meta_data(raw_meta_data: Any) -> dict:
        if isinstance(raw_meta_data, dict):
            return raw_meta_data
        if not raw_meta_data:
            return {}
        try:
            parsed = json.loads(raw_meta_data)
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _is_visual_node(node_data: dict) -> bool:
        entity_type = str(node_data.get("entity_type", "")).upper()
        if any(tag in entity_type for tag in ("IMAGE", "TABLE", "FORMULA")):
            return True
        meta_data = BaseGenerator.load_meta_data(node_data.get("meta_data"))
        return any(
            meta_data.get(key)
            for key in ("image_path", "img_path", "table_img_path", "equation_img_path")
        )

    @staticmethod
    def _extract_visual_asset_from_meta_data(meta_data: dict) -> str:
        if not isinstance(meta_data, dict):
            return ""
        for key in ("image_path", "img_path", "table_img_path", "equation_img_path"):
            asset_path = meta_data.get(key)
            if asset_path:
                return str(asset_path)
        return ""

    @classmethod
    def extract_visual_asset_path(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, _ = batch
        for _, node_data in nodes:
            if not cls._is_visual_node(node_data):
                continue
            meta_data = cls.load_meta_data(node_data.get("meta_data"))
            asset_path = cls._extract_visual_asset_from_meta_data(meta_data)
            if asset_path:
                return asset_path
        return ""

    async def generate(
        self,
        batch: tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ],
    ) -> list[dict]:
        """
        Generate QAs based on a given batch.
        :param batch
        :return: QA pairs
        """
        prompt = self.build_prompt(batch)
        image_path = self.extract_visual_asset_path(batch)
        response = await self.llm_client.generate_answer(
            prompt, image_path=image_path or None
        )
        qa_pairs = self.parse_response(response)  # generate one or more QA pairs
        return qa_pairs

    @staticmethod
    def format_vqa_generation_results(
        result: dict, output_data_format: str
    ) -> dict[str, Any]:
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
                    {"from": "human", "value": user_value},
                    {"from": "gpt", "value": [{"text": answer}]},
                ]
            }
        if output_data_format == "ChatML":
            user_content = [{"text": question}]
            if img_path:
                user_content[0]["image"] = img_path
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]
            }
        raise ValueError(f"Unknown output data format: {output_data_format}")

    @staticmethod
    def format_generation_results(
        result: dict, output_data_format: str
    ) -> dict[str, Any]:
        question = result.get("question", "")
        answer = result.get("answer", "")
        if "options" in result and result["options"]:
            options = result["options"]
            options_str = "\n".join(
                [f"{key}. {options[key]}" for key in sorted(options.keys())]
            )
            question += f"\nOptions:\n{options_str}"

        if output_data_format == "Alpaca":
            return {
                "instruction": question,
                "input": "",
                "output": answer,
            }

        if output_data_format == "Sharegpt":
            return {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            }
        if output_data_format == "ChatML":
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
        raise ValueError(f"Unknown output data format: {output_data_format}")
