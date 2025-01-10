# 引入必要的套件
import numpy as np  # 用於矩陣運算
from numpy import zeros, array, rot90  # 提供矩陣初始化與旋轉功能
import random  # 隨機數生成器
import matplotlib.pyplot as plt  # 繪圖工具
import math  # 數學運算模組
import torch  # PyTorch 深度學習框架
import torch.nn as nn  # 用於構建神經網絡的模組
import torch.optim as optim  # 優化器模組
import torch.nn.functional as F  # 常用神經網絡函數
import torchvision.transforms as T  # 圖像處理的工具
from collections import namedtuple, deque  # 用於儲存遊戲狀態的工具
from itertools import count  # 無窮迭代器
import os  # 用於文件操作

# 定義 Board 類，實現 2048 遊戲的核心邏輯
class Board():
    def __init__(self):
        # 初始化遊戲版面 (4x4)
        self.board = zeros((4, 4), dtype=int)
        self.fill_cell()  # 在隨機位置新增 2 或 4
        self.game_over = False  # 遊戲是否結束的狀態標誌
        self.total_score = 0  # 紀錄總得分

    def reset(self):
        # 重置遊戲版面
        self.__init__()

    def fill_cell(self):
        # 在空位上隨機填入數字 2 或 4
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            rnd = random.randint(0, i.size - 1) 
            self.board[i[rnd], j[rnd]] = 2 * ((random.random() > .9) + 1)

    def move_left(self, col):
        # 將一列方塊向左移動並合併相同數字的方塊
        new_col = zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0:
                if previous is None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        new_col[j] = 2 * col[i]  # 合併方塊
                        self.total_score += new_col[j]  # 更新得分
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous is not None:
            new_col[j] = previous
        return new_col

    def move(self, direction):
        # 根據方向 (0: 左, 1: 上, 2: 右, 3: 下) 移動遊戲版面
        rotated_board = rot90(self.board, direction)
        cols = [rotated_board[i, :] for i in range(4)]
        new_board = array([self.move_left(col) for col in cols])
        return rot90(new_board, -direction)

    def is_game_over(self):
        # 判斷遊戲是否結束
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 0:
                    return False
                if i != 0 and self.board[i - 1][j] == self.board[i][j]:
                    return False
                if j != 0 and self.board[i][j - 1] == self.board[i][j]:
                    return False
        return True

    def step(self, direction):
        # 執行一步操作，更新遊戲狀態
        new_board = self.move(direction)
        if not (new_board == self.board).all():
            self.board = new_board
            self.fill_cell()
# 輔助函式

def main_loop(b, direction):
    """
    執行遊戲的主要邏輯，根據指定方向移動棋盤：
    - 嘗試將棋盤移動到指定方向。
    - 如果棋盤未改變，則認為沒有移動。
    - 如果棋盤有變化，更新棋盤並新增隨機的數字格。
    - 返回移動是否成功。
    """
    new_board = b.move(direction)  # 嘗試移動棋盤
    moved = False
    if (new_board == b.board).all():  # 如果移動後的棋盤與原本相同
        pass
    else:
        moved = True
        b.board = new_board  # 更新棋盤
        b.fill_cell()  # 填充新的隨機格
    return moved

def encode_state(board):
    """
    將遊戲棋盤編碼為深度學習模型可以接受的格式：
    - 將棋盤中的數字轉換為對應的log2值 (例如 2 -> 1, 4 -> 2)，空格用0表示。
    - 使用 one-hot 編碼將每個數字轉為16維向量。
    - 將棋盤展平成一維，並重新調整為適合模型的輸入形狀 (NCHW)。
    """
    board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]  # log2編碼
    board_flat = torch.LongTensor(board_flat)  # 轉為 PyTorch 的 LongTensor
    board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()  # one-hot 編碼並展平
    board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)  # 調整形狀為 (N, C, H, W)
    return board_flat

# 裝置設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 檢查是否有可用的 GPU，否則使用 CPU

# 回放記憶庫
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
"""
Transition 是一個命名元組，用於儲存單次遊戲的經驗：
- state: 當前狀態
- action: 執行的動作
- next_state: 執行動作後的下一狀態
- reward: 執行該動作後獲得的回報
"""

class ReplayMemory(object):
    """
    回放記憶庫，儲存與管理遊戲過程中的經驗：
    - 使用一個固定大小的雙向佇列 (deque) 來儲存經驗。
    - 當記憶庫達到最大容量時，舊的經驗會自動被覆蓋。
    """

    def __init__(self, capacity):
        """
        初始化記憶庫：
        - capacity: 記憶庫的最大容量。
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """
        將一個遊戲經驗推入記憶庫。
        - args: 包含 state, action, next_state, reward 的參數。
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        隨機抽取指定數量的經驗樣本，用於訓練模型。
        - batch_size: 抽取的樣本數量。
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        返回記憶庫中儲存的經驗數量。
        """
        return len(self.memory)


# 定義神經網絡的結構，用於強化學習
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        # 使用多種卷積核大小來提取不同尺度的特徵
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        x = x.to(device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 2048)  # 第一層卷積
        self.conv2 = ConvBlock(2048, 2048)  # 第二層卷積
        self.conv3 = ConvBlock(2048, 2048)  # 第三層卷積
        self.dense1 = nn.Linear(2048 * 16, 1024)  # 全連接層
        self.dense6 = nn.Linear(1024, 4)  # 輸出層，對應四個動作

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)

# 訓練設置參數
BATCH_SIZE = 64  # 每次更新時從記憶庫中抽取的樣本數量
GAMMA = 0.99  # 折扣因子，用於計算未來回報的現值
EPS_START = 0.9  # 初始探索率
EPS_END = 0.01  # 最低探索率
EPS_DECAY = 0.9999  # 探索率隨時間衰減的因子
TARGET_UPDATE = 20  # 每多少回合更新一次目標網路
n_actions = 4  # 動作空間的數量

# 建立策略網路與目標網路
policy_net = DQN().to(device)  # 策略網路，負責實際選擇動作
target_net = DQN().to(device)  # 目標網路，提供穩定的目標值
target_net.load_state_dict(policy_net.state_dict())  # 初始化目標網路的參數與策略網路相同
target_net.eval()  # 設定目標網路為評估模式

optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)  # 使用Adam優化器更新策略網路的參數
memory = ReplayMemory(50000)  # 重放記憶庫，用於儲存遊戲過程中的經驗
steps_done = 0  # 記錄總執行步數

# 動作選擇
def select_action(state):
    """
    基於ε-貪婪策略選擇動作：
    - 以一定機率選擇隨機動作 (探索)。
    - 否則選擇模型計算的最佳動作 (利用)。
    """
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY ** steps_done))  # 計算當前探索率
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():  # 不需要計算梯度
            return policy_net(state).max(1)[1].view(1, 1)  # 選擇Q值最大的動作
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)  # 隨機選擇動作

# 模型優化
def optimize_model():
    """
    利用經驗回放記憶庫來進行策略網路的更新：
    - 計算當前Q值與目標Q值之間的損失。
    - 通過反向傳播來更新網路參數。
    """
    if len(memory) < BATCH_SIZE:  # 當記憶庫的資料不足時，不執行更新
        return
    transitions = memory.sample(BATCH_SIZE)  # 從記憶庫中抽取樣本
    batch = Transition(*zip(*transitions))  # 將樣本拆解為各欄位批次

    # 過濾非終止狀態，計算Q值與目標值
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 策略網路的當前Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # 目標網路的最大Q值
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # 計算目標Q值

    # 使用均方誤差損失函數計算損失並更新網路
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 儲存與載入模型
policy_net_path = "policy_net.pth"
target_net_path = "target_net.pth"

if os.path.isfile(policy_net_path):
    policy_net.load_state_dict(torch.load(policy_net_path))  # 載入策略網路參數
    print("策略網路載入成功。")
else:
    print("找不到已儲存的策略網路。將從頭開始訓練。")

if os.path.isfile(target_net_path):
    target_net.load_state_dict(torch.load(target_net_path))  # 載入目標網路參數
    print("目標網路載入成功。")
else:
    print("找不到已儲存的目標網路。將從頭開始訓練。")

# 主訓練迴圈
num_episodes = 3500  # 訓練的總回合數
total_scores, best_tile_list = [], []

for i_episode in range(num_episodes):
    print(f"回合 {i_episode}")
    game = Board()  # 初始化遊戲
    game.reset()  # 重置遊戲狀態
    state = encode_state(game.board).float()  # 將遊戲板狀態編碼為模型輸入
    non_valid_count, valid_count = 0, 0  # 記錄有效與無效移動次數

    for t in count():  # 每回合的遊戲循環
        action = select_action(state)  # 選擇動作
        old_score = game.total_score  # 紀錄執行動作前的得分
        game.step(action.item())  # 執行動作
        done = game.is_game_over()  # 檢查遊戲是否結束
        reward = torch.tensor([game.total_score - old_score], device=device)  # 計算即時回報

        if not done:
            next_state = encode_state(game.board).float()  # 更新遊戲狀態
        else:
            next_state = None

        if next_state is not None and torch.eq(state, next_state).all():  # 判斷是否為無效移動
            non_valid_count += 1
            reward -= 10  # 處罰無效移動
        else:
            valid_count += 1

        if next_state is None or len(memory) == 0 or not same_move(state, next_state, memory.memory[-1]):
            memory.push(state, action, next_state, reward)  # 將狀態與回報儲存至記憶庫
        
        state = next_state

        if done:
            for _ in range(100):
                optimize_model()  # 遊戲結束後進行多次模型優化
            print(game.board)
            print(f"回合得分: {game.total_score}")
            print(f"無效移動次數: {non_valid_count}")
            print(f"有效移動次數: {valid_count}")
            total_scores.append(game.total_score)
            best_tile_list.append(game.board.max())  # 紀錄遊戲中的最大數字
            if i_episode > 50:
                average = sum(total_scores[-50:]) / 50
                print(f"最近50回合平均分數: {average}")
            break

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())  # 定期更新目標網路
        policy_net.train()

    if i_episode % 100 == 0:
        torch.save(policy_net.state_dict(), policy_net_path)  # 定期儲存模型參數
        torch.save(target_net.state_dict(), target_net_path)
        print(f"已於第 {i_episode} 回合儲存模型。")

print("訓練完成，模型已儲存至本地端。")