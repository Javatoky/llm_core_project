from src.tools.errors import ToolExecutionError


async def get_weather(city: str) -> str:
    """查询天气"""
    city = city.strip()
    if not city:
        raise ToolExecutionError("城市名称不能为空")

    return f'{city}：晴，25°C'
