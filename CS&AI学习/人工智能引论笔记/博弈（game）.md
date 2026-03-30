~~为什么不能叫游戏~~

[[人工智能引论|回家（）]]

**博弈类型**：分类方式：确定还是随机？一个智能体还是多个？是否零和？是否是完美信息博弈？

- 一个问题建模：
	- 状态
	- 博弈主体
	- 行动
	- 转移方程
	- 结束测试
	- 最终效用
- 代价->效用
## 确定性博弈

### minimax
- 其实是一种[[搜索#^4c8be7|深搜]]，效率和深搜一致
- 假设对方决策最优
- 一种适用**多智能体**的推广：节点的价值和终止状态给出**元组**，每个玩家最大化自己的部分

*剩下的都知道了，不写了*

#### alpha-beta剪枝
*不写了*

[伪代码](https://inst.eecs.berkeley.edu/~cs188/textbook/games/minimax.html)

#### 评估函数
- 类似于启发

### expectimax
- 考虑不确定性
- 期望最大化 
- 可以采用混合层（多个minimum层，minimax＋随机层，等等）

### MCTS
- 总体思路：不断模拟，并从中**采样**，得到各个节点的估计的胜率，对胜率高的（推测这个节点更好）和探索次数少的进行探索（平衡预期好和不确定），选择两者兼优的节点
- 判断公式（UCB 最大置信）：$\text{UCT} = \frac{w_i}{n_i} + C \sqrt{\frac{\ln N}{n_i}}$
	- 其中w_i表示模拟的胜利次数，n_i表示总的模拟次数，N表示父节点的总模拟次数，C是一个权重，平衡**探索（增加广度，选择非最优的）** 和 **利用（增加深度，选择最优）**
	- $\frac{w_i}{n_i}$也可以替换为一种估值/期望，这也会影响到C
- epsilon-greedy算法：epsilon概率选择非最优的，1-epsilon概率选择最优的
- 算法：
	- 从根节点开始使用UCB算法层层向下，直到到达未探索过的节点
	- 加入新的节点，计算这个节点的胜率
	- 将胜利次数返回给之前的节点
	- 重复多次后，选择模拟次数最多的节点

（*突然感觉这个算法的计算量好多啊*）
伪代码
``` pseudo
function MCTS(root, time_limit):
    while within_time_limit(time_limit):
        # 1. 选择：从根开始，用UCT公式选最优点，直到叶子节点
        node = Select(root)
        
        # 2. 扩展：若节点未终止且未完全展开，添加一个子节点
        if not node.is_terminal():
            node = Expand(node)
        
        # 3. 模拟：从新节点开始随机对局（快速走子）
        reward = Simulate(node)
        
        # 4. 回溯：将模拟结果沿路径回传，更新胜场和访问次数
        Backpropagate(node, reward)
    
    # 最终落子：后期直接选访问次数最多的子节点（最大利用）
    return BestChild(root, 0)  # 参数0表示只利用不探索

# 选择阶段：使用UCT公式（C可随总模拟次数衰减）
function Select(node):
    while node.is_fully_expanded() and not node.is_terminal():
        node = BestChild(node, C)  # 动态C值：C = C_init * (1 - 进度比例)
    return node

# 最佳子节点选择：含探索/利用平衡
function BestChild(node, C):
    best_child = null
    best_value = -inf
    for child in node.children:
        # UCT公式
        exploit = child.wins / child.visits
        explore = C * sqrt(ln(node.visits) / child.visits)
        uct_value = exploit + explore
        if uct_value > best_value:
            best_value = uct_value
            best_child = child
    return best_child

# 扩展：选择一个未探索的动作，添加子节点
function Expand(node):
    action = node.unexplored_actions.pop()
    new_node = Node(state=node.state.apply(action))
    node.children.add(new_node)
    return new_node

# 模拟：随机走子至终局，返回胜负结果（1/0）
function Simulate(node):
    state = node.state
    while not state.is_terminal():
        action = random_legal_action(state)
        state = state.apply(action)
    return state.get_reward()  # 当前玩家视角的收益

# 回溯：更新路径上所有节点的统计信息
function Backpropagate(node, reward):
    while node is not null:
        node.visits += 1
        node.wins += reward  # reward需根据玩家视角转换符号
        reward = -reward      # 切换视角（零和博弈）
        node = node.parent
```


#### AlphaGo
通过先前的训练，可以用神经网络将投入的数据记录下来从而快速算出获胜概率
- 选择阶段，UCB第二项变为人类棋局训练得到的结果（获胜概率）
- 模拟：使用快速走子的策略代替随机模拟（会更好一点）
- 回溯的数值变为经验判断和模拟加权的估计

##### AlphaGo Zero
- 无需人类数据自我对弈
- 统一神经网络
