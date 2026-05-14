import asyncio

from src.adapters.model_adapter import LLMBackend, ModelConfig
from src.agents.hybrid_agent import hybrid_agent_reply
from src.config import (DS_API_KEY, DS_BASE_URL, DS_MODEL, QWEN_ALY_API_KEY,
                        QWEN_ALY_BASE_URL, QWEN_API_KEY, QWEN_BASE_URL,
                        QWEN_EMBED_MODEL, QWEN_MODEL)
from src.prompts.hybrid_system_prompt import HYBRID_SYSTEM_PROMPT

test_cases = [
    "你好，你能做什么？",
    "北京今天天气怎么样？",
    "帮我算一下 789 * 321",
    "Gemini 3提示词中给Gemini设定的Goal是什么？",
    "Trump最喜欢的食物是什么？",
    "根据知识库回答真正的友谊是什么，并顺便算一下 12 * 34"
]

COLLECTION_NAME = "my_documents"

def make_models(*configs: ModelConfig) -> list[LLMBackend]:
    models: list[LLMBackend] = []
    for config in configs:
        models.append(LLMBackend(config))
    return models

async def evaluate_model_result(backend: LLMBackend) -> tuple[str, list[tuple]]:
    """人工评测模型结果"""
    evaluations: list[tuple] = []
    model_name = backend.config.chat_model
    print(f"\n=== Model: {model_name} ===\n")
    for i, case in enumerate(test_cases, 1):
        messages = [
            {"role": "system", "content": HYBRID_SYSTEM_PROMPT}
        ]
        result = await hybrid_agent_reply(backend, case, messages, COLLECTION_NAME, True)
        print(f"[{i}]测试用例: {case}")
        print(f"模型输出：{result.answer}")
        print(f"Called tools: {result.called_tools}")
        print(f"Pass: (True/False)", end = '')
        ps = input()
        print(f"Notes: ", end = '')
        notes = input()
        print()
        evaluations.append((case, result.called_tools, ps, notes))

    return (model_name, evaluations)

def report_comparison(final_evaluations: list[tuple[str, list[tuple]]]):
    """最终评测结果呈现"""
    print(f"\n————最终结果————\n")
    for model_name, evaluations in final_evaluations:
        print(f"=== Model: {model_name} ===\n")
        for i, (case, called_tools, passed, notes) in enumerate(evaluations, 1):
            print(f"[{i}] {case}")
            print(f"Called tools: {called_tools}")
            print(f"Pass: {passed}")
            print(f"Notes: {notes}\n")
        print()

async def compare_model_performance():
    qwen_config = ModelConfig(
        chat_model = QWEN_MODEL,
        chat_api_key = QWEN_API_KEY,
        chat_base_url = QWEN_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    )
    deepseek_config = ModelConfig(
        chat_model = DS_MODEL,
        chat_api_key = DS_API_KEY,
        chat_base_url = DS_BASE_URL,
        embed_model = QWEN_EMBED_MODEL,
        embed_api_key = QWEN_ALY_API_KEY,
        embed_base_url = QWEN_ALY_BASE_URL,
        supports_embeddings = True
    )

    backends = make_models(qwen_config, deepseek_config)
    final_evaluations: list[tuple[str, list[tuple]]] = []
    for backend in backends:
        final_evaluations.append(await evaluate_model_result(backend))
    
    report_comparison(final_evaluations)

if __name__ == "__main__":
    asyncio.run(compare_model_performance())