def calculate_slope_and_intercept(x1, y1, x2, y2):
    """
    计算直线的斜率 k 和截距 b
    参数:
    x1, y1: 第一个点的坐标
    x2, y2: 第二个点的坐标

    返回:
    k: 斜率
    b: 截距
    """
    # 计算斜率 k
    if x2 == x1:  # 防止分母为零（垂直线）
        raise ValueError("直线是垂直的，斜率不存在")
    k = (y2 - y1) / (x2 - x1)

    # 计算截距 b
    b = y1 - k * x1  # 使用点 (x1, y1) 来计算 b

    return k, b


if __name__ == '__main__':
    pass

    x1, y1 = 0.2, -0.5
    x2, y2 = 0.6, 0.5

    k, b = calculate_slope_and_intercept(x1, y1, x2, y2)
    print(f"斜率 k = {k}, 截距 b = {b}")