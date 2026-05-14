# test_model_adapter_integration.py 测试方针

## 测试内容
集成测试：model_adapter_integration.py

测试覆盖内容：
至少写两个测试：

1. backend.chat() 能返回非空字符串
2. backend.embed(["你好"]) 能返回 list[list[float]]，且向量非空

建议加 pytest.mark.integration，以后可以区分 unit 和 integration：

@pytest.mark.integration
def test_chat_integration():
    ...


不要在 integration test 里断言具体回答内容，只断言结构：

assert isinstance(reply, str)
assert reply.strip()

Embedding 只断言：

assert isinstance(vectors, list)
assert len(vectors) == 1
assert isinstance(vectors[0], list)
assert len(vectors[0]) > 0
assert all(isinstance(x, float) for x in vectors[0][:10])
## 代码编写细则
1. 使用 pytest assert 测试，在 test_model_adapter_integration.py 中完善测试内容
2. 我并未安装异步 pytest，所以对于异步函数的测试你可以使用 asyncio.run() 运行，或尝试其他能在普通测试函数运行的方法
3. 我已在文件中导入相应配置，你使用注入的配置即可

## 问题补充
无
