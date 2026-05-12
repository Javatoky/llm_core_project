"""工具层异常类型"""


class ToolExecutionError(Exception):
    """工具执行失败"""
    pass


class ToolNotFoundError(Exception):
    """未找到工具"""
    pass


class ToolArgumentsError(Exception):
    """工具参数错误"""
    pass
