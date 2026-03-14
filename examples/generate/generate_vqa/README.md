# Generate VQAs

## DRAM-oriented high-quality VQA pipeline

This workflow is suitable for generating VQA training data from memory-system materials (e.g., DRAM timing diagrams, architecture figures, specs).

### 1) Prepare input
- Put your multimodal samples in JSON format (text + image path).
- Ensure each sample has enough textual context and image metadata so the graph builder can connect entities and relations.

### 2) Run generation
```bash
bash examples/generate/generate_vqa/generate_vqa.sh
```

### 3) Quality controls already enabled
- Prompt-level constraints for DRAM/VQA reasoning (structure, timing, performance, comparison, grounding).
- Post-generation filtering in `VQAGenerator`:
  - drop empty/too short/too long answers
  - drop uncertain answers (e.g., unknown)
  - deduplicate near-identical QA pairs
  - enforce context keyword grounding

### 4) Recommended config tuning
In `vqa_config.yaml` under `generate.params`, tune:
- `min_question_length`
- `min_answer_length`
- `max_answer_length`

For DRAM-heavy corpora, you can increase `min_question_length` slightly to reduce shallow questions.
