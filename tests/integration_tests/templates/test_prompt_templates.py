from graphgen.templates.kg.kg_extraction import KG_EXTRACTION_PROMPT
from graphgen.templates.kg.mm_kg_extraction import MMKG_EXTRACTION_PROMPT


def test_english_kg_extraction_template_has_no_chinese_markers():
    template = KG_EXTRACTION_PROMPT["en"]["TEMPLATE"]

    assert "输出：" not in template
    assert "示例" not in template
    assert "使用中文" not in template
    assert "中文" not in template


def test_english_kg_extraction_template_examples_keep_expected_record_shapes():
    template = KG_EXTRACTION_PROMPT["en"]["TEMPLATE"]

    assert '("entity"{tuple_delimiter}' in template
    assert '("relationship"{tuple_delimiter}' in template
    assert "<evidence_span>)" in template
    assert "<evidence_span>{tuple_delimiter}<confidence>)" in template


def test_english_mm_kg_extraction_template_uses_entity_without_evidence():
    template = MMKG_EXTRACTION_PROMPT["en"]

    assert '("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)' in template
    assert '("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>{tuple_delimiter}<evidence_span>)' not in template
    assert '("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<evidence_span>{tuple_delimiter}<confidence>)' in template
