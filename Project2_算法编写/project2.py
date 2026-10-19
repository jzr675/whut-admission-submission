import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import heapq
import os

# 设置中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 地图表示
test_map = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]# 定义列表"1"表示障碍物；"0"表示可通行区域
test_map = np.array(test_map)  # 转成 numPy 数组

# 笛卡尔坐标系下的起点、终点
start_point = (1, 0)
goal_point = (9, 9)

class Node:
    def __init__(self, x, y):
        self.x = x          # 栅格列索引
        self.y = y          # 栅格行索引
        self.g = 0.0        # 起点到当前节点的实际代价
        self.h = 0.0        # 当前节点到终点的启发估计代价
        self.f = 0.0        # 总优先级 f = g + h
        self.parent = None  # 父节点，用于回溯路径

    # 重载小于号，支持优先队列按f值从小到大排序
    def __lt__(self, other):
        return self.f < other.f

    # 重载等于号，按坐标判断两个节点是否为同一个格子
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# 三种启发式函数h
'''曼哈顿距离'''
def heuristic_manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

'''欧几里得距离'''
def heuristic_euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

'''切比雪夫距离'''
def heuristic_chebyshev(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)

# 两种扩展方式
'''四向扩展支持上下左右'''
DIRECTIONS_4 = [
    (-1, 0, 1.0), # 向左
    (1, 0, 1.0),  # 向右
    (0, -1, 1.0), # 向下
    (0, 1, 1.0)   # 向上
]

'''八向扩展方式支持对角'''
DIRECTIONS_8 = [
    (-1, 0, 1.0),  (1, 0, 1.0),  (0, -1, 1.0),  (0, 1, 1.0), # 向左, 右, 下, 上
    (-1, -1, math.sqrt(2)),  (1, -1, math.sqrt(2)),          # 左下, 右下
    (-1, 1, math.sqrt(2)),   (1, 1, math.sqrt(2))            # 右下, 右上
]

# A*算法主类
class AStarPlanner:
    def __init__(self, grid_map, heuristic_type = 'manhattan', direction_type = '4'):
        """
        初始化A*规划器
        grid_map: 二维numpy数组栅格地图，0=可通行，1=障碍物
        heuristic_type: 启发函数：manhattan / euclidean / chebyshev
        direction_type: 拓展方向：4 / 8
        """
        self.grid = grid_map
        self.height = grid_map.shape[0]    # 地图行数
        self.width = grid_map.shape[1]     # 地图列数

        # 选择启发函数
        if heuristic_type == 'manhattan':
            self.heuristic = heuristic_manhattan
        elif heuristic_type == 'euclidean':
            self.heuristic = heuristic_euclidean
        elif heuristic_type == 'chebyshev':
            self.heuristic = heuristic_chebyshev
        else:
            raise ValueError("启发函数输入错误，可选：manhattan / euclidean / chebyshev")

        # 选择拓展方向
        if direction_type == '4':
            self.directions = DIRECTIONS_4
        elif direction_type == '8':
            self.directions = DIRECTIONS_8
        else:
            raise ValueError("拓展方式输入错误，可选：4 / 8")

    def _cartesian_to_grid(self, x, y):
        '''笛卡尔坐标系转化为栅格坐标'''
        grid_x = x
        grid_y = self.height - 1 - y
        return grid_x, grid_y

    def _grid_to_cartesian(self, x, y):
        '''栅格坐标转化为笛卡尔坐标'''
        cart_x = x
        cart_y = self.height - 1 - y
        return cart_x, cart_y

    def is_valid(self, x, y):
        '''检查栅格坐标是否合法：在地图内且不是障碍物'''
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if self.grid[y, x] == 1:
            return False
        return True

    def plan(self, start, goal):
        """
        执行路径规划
        start: 起点笛卡尔坐标
        goal: 终点笛卡尔坐标
        """
        # 输入坐标转换：笛卡尔转化为栅格
        start_x, start_y = self._cartesian_to_grid(*start)
        goal_x, goal_y = self._cartesian_to_grid(*goal)

        # 起点终点合法性校验
        if not self.is_valid(start_x, start_y) or not self.is_valid(goal_x, goal_y):
            print("起点或终点越界或为障碍物")
            return []

        # 初始化起点节点
        start_node = Node(start_x, start_y)
        start_node.h = self.heuristic(start_x, start_y, goal_x, goal_y)
        start_node.f = start_node.g + start_node.h

        # 初始化Open表和Close表
        open_heap = []
        heapq.heappush(open_heap, start_node)
        close_set = set()

        # 主搜索循环
        while open_heap:
            # 取出f值最小的节点
            current_node = heapq.heappop(open_heap)
            current_pos = (current_node.x, current_node.y)

            # 若该节点已探索过则跳过
            if current_pos in close_set:
                continue

            # 到达终点回溯路径并转换为笛卡尔坐标返回
            if current_node.x == goal_x and current_node.y == goal_y:
                grid_path = self._reconstruct_path(current_node)
                cart_path = [self._grid_to_cartesian(x, y) for x, y in grid_path]
                return cart_path

            # 标记为已探索
            close_set.add(current_pos)

            # 遍历所有拓展方向，生成邻居节点
            for dx, dy, cost in self.directions:
                neighbor_x = current_node.x + dx
                neighbor_y = current_node.y + dy

                # 邻居不合法或已探索则跳过
                if not self.is_valid(neighbor_x, neighbor_y):
                    continue
                if (neighbor_x, neighbor_y) in close_set:
                    continue

                # 计算新的代价并生成邻居节点
                new_g = current_node.g + cost
                neighbor_node = Node(neighbor_x, neighbor_y)
                neighbor_node.g = new_g
                neighbor_node.h = self.heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                neighbor_node.parent = current_node

                # 加入Open表
                heapq.heappush(open_heap, neighbor_node)

        # Open表为空，无可行路径
        print("未找到可行路径")
        return []

    def _reconstruct_path(self, end_node):
        '''从终点回溯父节点，得到栅格坐标路径'''
        path = []
        current = end_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()  # 反转成起点终点顺序
        return path


# matplotlib 可视化函数
def plot_astar_result(grid, path):
    """
    可视化A*规划结果
    grid: 栅格地图
    path: 笛卡尔坐标路径列表
    """
    height = len(grid)
    width = len(grid[0])

    # 地图上下翻转，适配笛卡尔坐标系
    display_grid = [row.copy() for row in grid] # 复制原地图
    display_grid = display_grid[::-1]           # 整体进行翻转

    plt.figure(figsize=(8, 8))

    # 绘制栅格底图
    plt.imshow(
        display_grid,
        cmap='gray_r',  # 白色=可通行，黑色=障碍物
        origin='lower',
        extent=[-0.5, width-0.5, -0.5, height-0.5]  # 整数坐标对齐格子中心
    )

    # 提取路径坐标
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    # 绘制路径线和路径点
    plt.plot(path_x, path_y, color='cornflowerblue', linewidth=3, label='规划路径')
    plt.scatter(path_x, path_y, color='cornflowerblue', s=60)

    # 标记起点和终点
    plt.scatter(path_x[0], path_y[0], color='limegreen', s=200, zorder=5, label='起点 S')
    plt.scatter(path_x[-1], path_y[-1], color='crimson', s=200, zorder=5, label='终点 G')

    # 图表设置
    plt.xticks(range(width))
    plt.yticks(range(height))
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.title('A* 栅格路径规划结果')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.axis('equal')

# 保存图片
    save_folder = r"E:\作业\课题组面试"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = os.path.join(save_folder, "astar_path_chebyshev+8")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")

    plt.show()


# 测试主程序
if __name__ == "__main__":

    # 创建A*规划器，可修改参数测试
    planner = AStarPlanner(
        grid_map=test_map,
        heuristic_type='chebyshev',  # manhattan / euclidean / chebyshev
        direction_type='8'           # 4 / 8
    )

    # 执行路径规划
    result_path = planner.plan(start_point, goal_point)

    # 输出结果并可视化
    if result_path:
        print(f"找到路径，共 {len(result_path)} 个点（笛卡尔坐标）：")
        print(result_path)
        # 弹出可视化窗口
        plot_astar_result(test_map, result_path)
    else:
        print("规划失败，无可行路径")