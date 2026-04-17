# pylint: disable=C0301
TEMPLATE_ZH: str = """请基于以下知识子图生成多跳推理问题和答案。你将获得一个知识子图，其中包含一系列实体、关系和事实。
你的任务是生成一个问答对，其中问题需要经过多次推理才能回答。问题的答案应该是从给定的知识子图中推断出来的。确保问题的难度适中，需要多步推理才能回答。

请注意下列要求：
1. 仅输出一个问答（QA）对，不得包含任何额外说明或分析
2. 不得重复答案内容或其中任何片段，不要直接复制示例问题和答案
3. 答案应准确且直接从文本中得出。确保QA对与给定文本的主题或重要细节相关。

输出格式：
<question>question_text</question>
<answer>answer_text</answer>

例如：
输入为：
--实体--
1. 苹果
2. 水果
3. 维生素C
--关系--
1. 苹果-水果：苹果是一种水果
2. 水果-维生素C：水果中富含维生素C

输出：
<question>通过吃苹果补充的什么物质，有助于维持健康？</question>
<answer>维生素C</answer>

真实输入如下：
--实体--
{entities}
--关系--
{relationships}

输出：
"""

TEMPLATE_EN: str = """Please generate a multi-hop reasoning question and answer based on the following knowledge subgraph. You will be provided with a knowledge subgraph that contains a series of entities, relations, and facts.
Your task is to generate a question-answer (QA) pair where the question requires multiple steps of reasoning to answer. The answer to the question should be inferred from the given knowledge subgraph. Ensure that the question is of moderate difficulty and requires multiple steps of reasoning to answer.

Please note the following requirements:
1. Output only one QA pair without any additional explanations or analysis.
2. Do not repeat the content of the answer or any part of it. Do not directly copy the example question and answer.
3. The answer should be accurate and directly derived from the text. Make sure the QA pair is relevant to the main theme or important details of the given text.

For example:
Input:
--Entities--
1. Apple
2. Fruit
3. Vitamin C
--Relations--
1. Apple-Fruit: Apple is a type of fruit
2. Fruit-Vitamin C: Fruits are rich in Vitamin C

Output:
<question>What substance, obtained by eating apples, helps maintain health?</question>
<answer>Vitamin C</answer>

Real input:
--Entities--
{entities}
--Relations--
{relationships}

Output:
"""

MULTI_HOP_GENERATION_PROMPT = {"en": TEMPLATE_EN, "zh": TEMPLATE_ZH}

SOURCE_CONTEXT_SECTION_ZH: str = """补充证据规则：
下方原始来源片段是对应实体的上下文证据。请围绕片段内容中的事实、数值、过程、比较和关系构造多跳推理链。`Source:` 标签只作为轻量来源标记。不要围绕来源名、文件名、chunk id 或元数据提问。不要引入实体、关系或来源片段均不支持的信息。

--原始来源片段--
{source_chunks}
"""

SOURCE_CONTEXT_SECTION_EN: str = """Additional grounding rule:
The original source chunks below are contextual evidence for the listed entities. Build the multi-hop reasoning chain from facts, measurements, procedures, comparisons, and relationships stated in the chunk content. Treat `Source:` labels only as lightweight provenance labels. Do not ask about source names, file names, chunk ids, or metadata. Do not introduce facts unsupported by the entities, relations, or source chunks.

--Original Source Chunks--
{source_chunks}
"""

TEMPLATE_WITH_SOURCE_CONTEXT_ZH: str = TEMPLATE_ZH.replace(
    "\n输出：",
    f"\n{SOURCE_CONTEXT_SECTION_ZH}\n输出：",
)

TEMPLATE_WITH_SOURCE_CONTEXT_EN: str = TEMPLATE_EN.replace(
    "\nOutput:",
    f"\n{SOURCE_CONTEXT_SECTION_EN}\nOutput:",
)

MULTI_HOP_GENERATION_PROMPT_WITH_SOURCE_CONTEXT = {
    "en": TEMPLATE_WITH_SOURCE_CONTEXT_EN,
    "zh": TEMPLATE_WITH_SOURCE_CONTEXT_ZH,
}
