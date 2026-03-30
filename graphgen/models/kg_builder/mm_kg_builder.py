import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

from graphgen.bases import Chunk
from graphgen.templates import MMKG_EXTRACTION_PROMPT
from graphgen.utils import (
    detect_main_language,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    logger,
    normalize_evidence_text,
    split_string_by_multi_markers,
)

from .light_rag_kg_builder import LightRAGKGBuilder


class MMKGBuilder(LightRAGKGBuilder):
    OCR_MAX_CHARS = 1600
    OCR_MIN_FRAGMENT_LEN = 3

    @staticmethod
    def _resolve_payload(chunk: Chunk) -> dict:
        metadata = dict(chunk.metadata or {})
        payload = {}

        content = chunk.content
        if isinstance(content, dict):
            payload.update(content)
        elif isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = json.loads(stripped)
                except Exception:  # pylint: disable=broad-except
                    parsed = None
                if isinstance(parsed, dict):
                    payload.update(parsed)

        for key, value in metadata.items():
            payload.setdefault(key, value)

        return payload

    @staticmethod
    def _coerce_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            parts = [normalize_evidence_text(str(item)) for item in value]
            return "\n".join(part for part in parts if part)
        return normalize_evidence_text(str(value))

    @classmethod
    def _dedupe_lines(cls, text: str) -> str:
        seen = set()
        lines = []
        for raw_line in str(text or "").splitlines():
            line = normalize_evidence_text(raw_line)
            if not line:
                continue
            lowered = line.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            lines.append(line)
        return "\n".join(lines)

    @classmethod
    def _normalize_ocr_text(cls, text: str) -> str:
        kept = []
        for raw_line in str(text or "").splitlines():
            line = normalize_evidence_text(raw_line)
            if not line:
                continue
            if len(line) < cls.OCR_MIN_FRAGMENT_LEN:
                continue
            if re.fullmatch(r"(page|p\.)?\s*\d+\s*(/\s*\d+)?", line.lower()):
                continue
            if re.fullmatch(r"[\d\s,.;:()/-]+", line):
                continue
            if re.fullmatch(r"[xy]\s*=\s*[-+]?\d+(?:\.\d+)?", line.lower()):
                continue
            kept.append(line)

        normalized = cls._dedupe_lines("\n".join(kept))
        if len(normalized) <= cls.OCR_MAX_CHARS:
            return normalized

        clipped_lines = []
        current_len = 0
        for line in normalized.splitlines():
            next_len = current_len + len(line) + (1 if clipped_lines else 0)
            if next_len > cls.OCR_MAX_CHARS:
                break
            clipped_lines.append(line)
            current_len = next_len
        return "\n".join(clipped_lines)

    async def _extract_image_ocr_text(self, image_path: str) -> str:
        if not image_path:
            return ""

        prompt = (
            "Extract the visible technical text from this image.\n"
            "Return concise plain text only.\n"
            "Include OCR text, labels, legends, axis titles, table-like headings, "
            "measurement values, units, and short technical phrases that are visibly present.\n"
            "Ignore decorative content, repeated boilerplate, page furniture, and speculation.\n"
            "Prefer one fact or label per line."
        )
        try:
            result = await self.llm_client.generate_answer(prompt, image_path=image_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Image OCR extraction failed for %s: %s", image_path, exc)
            return ""
        return self._normalize_ocr_text(result)

    async def _build_image_evidence_pack(
        self, chunk: Chunk, metadata: dict
    ) -> tuple[str, dict]:
        metadata = dict(metadata or {})

        image_caption = self._coerce_text(metadata.get("image_caption"))
        note_text = self._coerce_text(metadata.get("note_text"))
        ocr_text = self._normalize_ocr_text(self._coerce_text(metadata.get("ocr_text")))

        image_path = self._coerce_text(
            metadata.get("img_path") or metadata.get("image_path")
        )
        if not ocr_text and image_path:
            ocr_text = await self._extract_image_ocr_text(image_path)
            if ocr_text:
                metadata["ocr_text"] = ocr_text

        sections = []
        if image_caption:
            sections.append(f"[Image Caption]\n{image_caption}")
        if note_text:
            sections.append(f"[Nearby Text]\n{note_text}")
        if ocr_text:
            sections.append(f"[OCR]\n{ocr_text}")

        combined = "\n\n".join(sections).strip()
        fallback_content = self._coerce_text(chunk.content)
        if (not combined or len(combined) < 40) and fallback_content:
            if combined:
                sections.append(f"[Fallback Context]\n{fallback_content}")
            else:
                sections.append(fallback_content)
            combined = "\n\n".join(section for section in sections if section).strip()

        return combined, metadata

    @staticmethod
    def _parse_records(result: str):
        records = split_string_by_multi_markers(
            result,
            [
                MMKG_EXTRACTION_PROMPT["FORMAT"]["record_delimiter"],
                MMKG_EXTRACTION_PROMPT["FORMAT"]["completion_delimiter"],
            ],
        )

        nodes = defaultdict(list)
        edges = defaultdict(list)
        return records, nodes, edges

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """
        Extract entities and relationships from a single multi-modal chunk using the LLM client.
        Expect to get a mini graph which contains a central multi-modal entity
        and its related text entities and relationships.
        Like:
        (image: "image_of_eiffel_tower") --[located_in]--> (text: "Paris")
        (image: "image_of_eiffel_tower") --[built_in]--> (text: "1889")
        (text: "Eiffel Tower") --[height]--> (text: "324 meters")
        :param chunk
        """
        chunk_id = chunk.id
        chunk_type = chunk.type  # image | table | formula | ...
        metadata = self._resolve_payload(chunk)

        # choose different extraction strategies based on chunk type
        if chunk_type == "image":
            image_text, metadata = await self._build_image_evidence_pack(chunk, metadata)
            language = detect_main_language(image_text or chunk.content)
            prompt_template = MMKG_EXTRACTION_PROMPT[language].format(
                **MMKG_EXTRACTION_PROMPT["FORMAT"],
                chunk_type=chunk_type,
                chunk_id=chunk_id,
                chunk_text=image_text or chunk.content,
                modality_guidance=MMKG_EXTRACTION_PROMPT["IMAGE_GUIDANCE"][language],
            )
            result = await self.llm_client.generate_answer(prompt_template)
            logger.debug("Image chunk extraction result: %s", result)
            records, nodes, edges = self._parse_records(result)

            for record in records:
                match = re.search(r"\((.*)\)", record)
                if not match:
                    continue
                inner = match.group(1)

                attributes = split_string_by_multi_markers(
                    inner, [MMKG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
                )

                entity = await handle_single_entity_extraction(attributes, chunk_id)
                if entity is not None:
                    if entity["entity_type"] == "IMAGE":
                        entity["metadata"] = metadata
                    nodes[entity["entity_name"]].append(entity)
                    continue

                relation = await handle_single_relationship_extraction(
                    attributes, chunk_id
                )
                if relation is not None:
                    key = (relation["src_id"], relation["tgt_id"])
                    edges[key].append(relation)

            return dict(nodes), dict(edges)

        if chunk_type == "table":
            table_caption = "\n".join(metadata.get("table_caption", []))
            table_body = metadata.get("table_body", "")
            table_text = "\n\n".join(
                part for part in [table_caption, table_body] if str(part).strip()
            )
            language = detect_main_language(table_text or chunk.content)
            prompt_template = MMKG_EXTRACTION_PROMPT[language].format(
                **MMKG_EXTRACTION_PROMPT["FORMAT"],
                chunk_type=chunk_type,
                chunk_id=chunk_id,
                chunk_text=table_text or chunk.content,
                modality_guidance="",
            )
            result = await self.llm_client.generate_answer(prompt_template)
            logger.debug("Table chunk extraction result: %s", result)

            records, nodes, edges = self._parse_records(result)

            for record in records:
                match = re.search(r"\((.*)\)", record)
                if not match:
                    continue
                inner = match.group(1)

                attributes = split_string_by_multi_markers(
                    inner, [MMKG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
                )

                entity = await handle_single_entity_extraction(attributes, chunk_id)
                if entity is not None:
                    if entity["entity_type"] == "TABLE":
                        entity["metadata"] = metadata
                    nodes[entity["entity_name"]].append(entity)
                    continue

                relation = await handle_single_relationship_extraction(
                    attributes, chunk_id
                )
                if relation is not None:
                    key = (relation["src_id"], relation["tgt_id"])
                    edges[key].append(relation)

            return dict(nodes), dict(edges)
        if chunk_type == "formula":
            pass  # TODO: implement formula-based entity and relationship extraction

        logger.error("Unsupported chunk type for MMKGBuilder: %s", chunk_type)
        return defaultdict(list), defaultdict(list)
