# 必要的函式庫導入
import torch
import torch.nn.functional as F
import math
import time
import cv2
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from torch.autograd import Variable
import numpy as np
import hopenet
import torchvision
import utils

class Board():
    def __init__(self):
        """
        初始化遊戲棋盤：
        - 建立一個 4x4 的零矩陣作為棋盤。
        - 在棋盤上隨機填入一個初始數字（2 或 4）。
        - 初始化遊戲狀態 (game_over) 為 False。
        - 初始化總分 (total_score) 為 0。
        """
        self.board = np.zeros((4, 4), dtype=int)
        self.fill_cell()
        self.game_over = False
        self.total_score = 0
    
    def reset(self):
        """
        重置遊戲：
        - 重新初始化遊戲棋盤和狀態。
        """
        self.__init__()
    
    def fill_cell(self):
        """
        在棋盤上隨機選擇一個空格填入 2 或 4：
        - 90% 的機率填入 2，10% 的機率填入 4。
        """
        i, j = np.where(self.board == 0)  # 找到所有空格的位置
        if i.size > 0:
            rnd = np.random.randint(0, i.size)  # 隨機選擇一個空格
            self.board[i[rnd], j[rnd]] = 2 * ((np.random.random() > .9) + 1)  # 填入數字
    
    def move_left(self, col):
        """
        將一列數字向左移動並合併相同的數字：
        - 將非零數字移到左側。
        - 若相鄰的數字相同，則合併為其兩倍，並累加得分。
        """
        new_col = np.zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0:  # 當前數字不為零
                if previous == None:
                    previous = col[i]  # 儲存目前的數字
                else:
                    if previous == col[i]:  # 相鄰數字相同
                        new_col[j] = 2 * col[i]  # 合併數字
                        self.total_score += new_col[j]  # 累加得分
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous  # 將之前的數字移入新列
                        j += 1
                        previous = col[i]
        if previous != None:  # 將最後一個數字移入新列
            new_col[j] = previous
        return new_col

    def move(self, direction):
        """
        根據指定方向移動棋盤：
        - 方向 0: 左移
        - 方向 1: 上移
        - 方向 2: 右移
        - 方向 3: 下移
        """
        rotated_board = np.rot90(self.board, direction)  # 旋轉棋盤以簡化邏輯
        cols = [rotated_board[i, :] for i in range(4)]
        new_board = np.array([self.move_left(col) for col in cols])  # 處理每一列
        return np.rot90(new_board, -direction)  # 旋轉回原始方向
    
    def is_game_over(self):
        """
        判斷遊戲是否結束：
        - 若棋盤有空格或相鄰數字相同，遊戲尚未結束。
        """
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 0:  # 存在空格
                    return False
                if i != 0 and self.board[i - 1][j] == self.board[i][j]:  # 上方相同
                    return False
                if j != 0 and self.board[i][j - 1] == self.board[i][j]:  # 左側相同
                    return False
        return True

    def step(self, direction):
        """
        執行一步遊戲動作：
        - 根據指定方向移動棋盤。
        - 若棋盤有變化，新增隨機數字到棋盤。
        - 更新遊戲是否結束的狀態。
        """
        new_board = self.move(direction)
        if not (new_board == self.board).all():  # 若棋盤有變化
            self.board = new_board
            if not self.is_game_over():  # 若遊戲尚未結束
                self.fill_cell()
        self.game_over = self.is_game_over()

    def draw_board(self, frame):
        """
        繪製棋盤到指定的框架上：
        - 使用背景圖作為基底。
        - 根據棋盤數字繪製對應的圖片或空格。
        """
        board_size = 700  # 棋盤總尺寸
        cell_size = board_size // 4  # 每個格子的大小
        margin = 10  # 格子間的間隙
        inner_cell_size = cell_size - margin  # 調整格子內部尺寸
        board_img = cv2.imread('picture/background.jpg')  # 背景圖片
        board_img = cv2.resize(board_img, (board_size, board_size))  # 調整背景大小

        # 載入數字圖片
        tile_images = {
            2: cv2.imread('picture/2.jpg'),
            4: cv2.imread('picture/4.jpg'),
            8: cv2.imread('picture/8.jpg'),
            16: cv2.imread('picture/16.jpg'),
            32: cv2.imread('picture/32.jpg'),
            64: cv2.imread('picture/64.jpg'),
            128: cv2.imread('picture/128.jpg'),
            256: cv2.imread('picture/256.jpg'),
            512: cv2.imread('picture/512.jpg'),
            1024: cv2.imread('picture/1024.jpg'),
            2048: cv2.imread('picture/2048.jpg'),
            4096: cv2.imread('picture/4096.jpg'),
        }

        # 調整數字圖片大小
        for value, img in tile_images.items():
            tile_images[value] = cv2.resize(img, (inner_cell_size, inner_cell_size))

        # 繪製每個格子
        for i in range(4):
            for j in range(4):
                value = self.board[i, j]
                x = j * cell_size + margin // 2  # 計算格子的 x 座標
                y = i * cell_size + margin // 2  # 計算格子的 y 座標

                if value in tile_images:  # 若有對應的數字圖片
                    tile_img = tile_images[value]
                    board_img[y:y + inner_cell_size, x:x + inner_cell_size] = tile_img
                else:
                    # 繪製空格背景
                    cv2.rectangle(board_img, (x, y), (x + inner_cell_size, y + inner_cell_size), (183, 195, 206), -1)

        # 將棋盤覆蓋到框架上
        frame[0:board_size, 0:board_size] = board_img[:board_size, :board_size]


class ConvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, device):
        """
        初始化卷積塊：
        - input_dim: 輸入通道數
        - output_dim: 輸出通道數
        - device: 設備（如 'cuda' 或 'cpu'）
        """
        super(ConvBlock, self).__init__()
        self.device = device
        d = output_dim // 4  # 將輸出通道數均分為四部分
        self.conv1 = torch.nn.Conv2d(input_dim, d, 1, padding='same')  # 1x1 卷積
        self.conv2 = torch.nn.Conv2d(input_dim, d, 2, padding='same')  # 2x2 卷積
        self.conv3 = torch.nn.Conv2d(input_dim, d, 3, padding='same')  # 3x3 卷積
        self.conv4 = torch.nn.Conv2d(input_dim, d, 4, padding='same')  # 4x4 卷積

    def forward(self, x):
        """
        前向傳播：
        - 將輸入數據分別通過四種不同的卷積核處理。
        - 將結果在通道維度上拼接。
        """
        x = x.to(self.device)  # 將數據移至指定設備
        output1 = self.conv1(x)  # 1x1 卷積
        output2 = self.conv2(x)  # 2x2 卷積
        output3 = self.conv3(x)  # 3x3 卷積
        output4 = self.conv4(x)  # 4x4 卷積
        return torch.cat((output1, output2, output3, output4), dim=1)  # 通道維度拼接

class DQN(torch.nn.Module):
    def __init__(self, device):
        """
        初始化 DQN 模型：
        - device: 設備（如 'cuda' 或 'cpu'）
        """
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = ConvBlock(16, 2048, device)  # 第一個卷積塊，輸入通道數為 16（4x4x4）
        self.conv2 = ConvBlock(2048, 2048, device)  # 第二個卷積塊
        self.conv3 = ConvBlock(2048, 2048, device)  # 第三個卷積塊
        self.dense1 = torch.nn.Linear(2048 * 16, 1024)  # 全連接層，將展平的特徵壓縮到 1024 維
        self.dense6 = torch.nn.Linear(1024, 4)  # 最後輸出層，對應四個動作

    def forward(self, x):
        """
        前向傳播：
        - 將輸入數據依次通過卷積塊和全連接層，最終輸出動作值。
        """
        x = x.to(self.device)  # 將數據移至指定設備
        x = F.relu(self.conv1(x))  # 激活函數
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.nn.Flatten()(x)  # 展平特徵
        x = F.dropout(self.dense1(x))  # 全連接層，並添加 Dropout 防止過擬合
        return self.dense6(x)  # 最終輸出動作值（Q 值）

def load_pose_model(snapshot_path):
    """
    加載頭部姿態模型
    - snapshot_path: 預訓練模型的路徑
    """
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)  # 初始化 Hopenet 模型
    saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cuda'))  # 加載預訓練權重
    model.load_state_dict(saved_state_dict)  # 設置模型的權重
    model.eval()  # 設置模型為評估模式
    return model

def load_dqn_model(model_path, device):
    """
    加載 DQN 模型
    - model_path: 訓練好的 DQN 模型的路徑
    - device: 設備 ('cpu' 或 'cuda')
    """
    model = DQN(device).to(device)  # 初始化 DQN 模型並將其移到指定設備
    checkpoint = torch.load(model_path, map_location=device)  # 加載模型檔案
    model.load_state_dict(checkpoint)  # 設置模型的權重
    model.eval()  # 設置模型為評估模式
    return model

def load_policy_model(model_path, device):
    """
    加載策略模型（DQN）
    - model_path: 策略模型的路徑
    - device: 設備 ('cpu' 或 'cuda')
    """
    model = DQN(device).to(device)  # 初始化 DQN 模型並將其移到指定設備
    checkpoint = torch.load(model_path, map_location=device)  # 加載模型檔案
    model.load_state_dict(checkpoint)  # 設置模型的權重
    model.eval()  # 設置模型為評估模式
    return model

def load_target_model(model_path, device):
    """
    加載目標模型（與策略模型相同架構）
    - model_path: 目標模型的路徑
    - device: 設備 ('cpu' 或 'cuda')
    """
    target_model = DQN(device).to(device)  # 初始化目標模型並將其移到指定設備
    checkpoint = torch.load(model_path, map_location=device)  # 加載模型檔案
    target_model.load_state_dict(checkpoint)  # 設置模型的權重
    target_model.eval()  # 設置模型為評估模式
    return target_model

def predict_move_with_model(model, state, device):
    """
    使用模型預測下一步的動作
    - model: 訓練好的模型
    - state: 當前遊戲狀態
    - device: 設備 ('cpu' 或 'cuda')
    """
    state = encode_state(state).float().to(device)  # 將遊戲狀態編碼並轉換為浮點數
    with torch.no_grad():  # 禁用梯度計算以加快預測
        action = model(state).max(1)[1].view(1, 1)  # 預測動作，選擇最大 Q 值對應的動作
    return action.item()  # 返回預測的動作

# 編碼遊戲狀態的函數
def encode_state(board):
    """
    將遊戲棋盤狀態編碼為神經網絡可以處理的格式
    - board: 4x4 的遊戲棋盤
    """
    # 將棋盤上的每個數字轉換為對應的對數（2 的冪次方）
    board_flat = [0 if e == 0 else int(math.log(e, 2)) for e in board.flatten()]
    board_flat = torch.LongTensor(board_flat)  # 轉換為 LongTensor 格式
    board_flat = F.one_hot(board_flat, num_classes=16).float().flatten()  # 將數字進行 one-hot 編碼
    board_flat = board_flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)  # 重新調整形狀，符合卷積層的要求
    return board_flat


class HeadControl:
    def __init__(self, debounce_time=0.5, pose_threshold=20):
        """
        初始化頭部控制器
        - debounce_time: 觸發間隔時間，防止過於頻繁的觸發
        - pose_threshold: 頭部姿態變化的閾值，用來判斷是否觸發動作
        """
        self.debounce_time = debounce_time  # 設定有效觸發的時間間隔
        self.pose_threshold = pose_threshold  # 設定頭部姿態變化的閾值
        self.last_trigger_time = time.time()  # 上次觸發的時間
        self.last_head_position = None  # 上次的頭部姿勢

    def should_trigger_move(self, current_head_position):
        """
        根據當前頭部姿態來判斷是否應該觸發移動。
        只有當頭部姿態有顯著變化（大於設定閾值）時，才會觸發。
        """
        current_time = time.time()  # 當前時間
        if current_head_position != self.last_head_position:  # 如果當前頭部姿勢與上次不同
            if current_time - self.last_trigger_time > self.debounce_time:  # 如果觸發間隔超過設定時間
                self.last_trigger_time = current_time  # 更新上次觸發時間
                self.last_head_position = current_head_position  # 更新頭部姿勢
                return True  # 觸發移動
        return False  # 不觸發移動

# 函數根據頭部的俯仰角、偏航角和滾動角來分類頭部姿態
def classify_head_pose(yaw, pitch, roll, pose_threshold=25):
    """
    根據頭部的偏航角、俯仰角和滾動角來分類頭部姿勢，
    當偏航角、俯仰角或滾動角超過某個閾值時，進行分類。
    """
    if pitch > pose_threshold:  # 俯仰角大於閾值，表示頭部抬起
        return "Head Up"
    elif pitch < -pose_threshold:  # 俯仰角小於負閾值，表示頭部低下
        return "Head Down"
    elif yaw > pose_threshold - 5:  # 偏航角大於閾值，表示頭部向右
        return "Head Right"
    elif yaw < -pose_threshold + 3:  # 偏航角小於負閾值，表示頭部向左
        return "Head Left"
    elif roll > pose_threshold - 5:  # 滾動角大於閾值，表示頭部向右傾斜
        return "Tilt Right"
    elif roll < -pose_threshold + 3:  # 滾動角小於負閾值，表示頭部向左傾斜
        return "Tilt Left"
    else:
        return "Neutral"  # 頭部保持中立，無明顯變化


# 主程式迴圈
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 設定運行設備，選擇CUDA（GPU）或CPU

    cap = cv2.VideoCapture(0)  # 開啟攝影機
    if not cap.isOpened():
        print("Error: Could not open camera.")  # 若無法開啟攝影機，顯示錯誤訊息
        return

    game = Board()  # 初始化遊戲物件
    pose_model = load_pose_model("300w_lp.pkl").to(device)  # 載入頭部姿勢模型
    yolo_model = YOLO('yolov11s-face.pt')  # 載入YOLO人臉檢測模型
    yolo_model.conf = 0.5  # 設定YOLO檢測的信心度閾值

    # 載入AI控制策略模型
    policy_model_path = "policy_net.pth"
    target_model_path = "target_net.pth"
    policy_model = load_policy_model(policy_model_path, device)  # 載入策略模型
    target_model = load_target_model(target_model_path, device)  # 載入目標模型
    dqn_model = load_dqn_model(policy_model_path, device)  # 載入DQN模型

    # 載入顯示圖片（控制模式圖片）
    mode_image_x = 116
    mode_image_y = 150
    mode_image_width = 175
    mode_image_height = 113
    score_image_x = 304
    score_image_y = 150
    score_image_width = 325
    score_image_height = 113
    AI_mode_img = cv2.imread('picture/AI.jpg')
    AI_mode_img = cv2.resize(AI_mode_img, (mode_image_width, mode_image_height)) 
    Head_mode_img = cv2.imread('picture/Head.jpg')
    Head_mode_img = cv2.resize(Head_mode_img, (mode_image_width, mode_image_height)) 
    WASD_mode_img = cv2.imread('picture/WASD.jpg')
    WASD_mode_img = cv2.resize(WASD_mode_img, (mode_image_width, mode_image_height)) 
    Score_img = cv2.imread('picture/Score.jpg')
    Score_img = cv2.resize(Score_img, (score_image_width, score_image_height)) 

    transformations = transforms.Compose([  # 圖像轉換（用於頭部姿勢識別）
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)  # 頭部姿勢的索引張量

    last_key_time = time.time()  # 上次按鍵時間
    debounce_time = 1  # 防抖時間間隔（防止重複按鍵）
    head_control = HeadControl(debounce_time)  # 頭部控制實例

    # 初始控制模式為'頭部姿勢'
    use_head_pose = True
    use_ai_control = False  # AI控制初始為關閉
    last_mode_switch_time = time.time()  # 記錄上次模式切換的時間

    # 設置AI控制的遊玩速度
    ai_playing_speed = 0.1  # 設定AI遊玩速度（以秒為單位）
    last_ai_move_time = time.time()

    while True:
        ret, frame = cap.read()  # 讀取攝影機畫面
        if not ret:
            break  # 若讀取失敗，退出循環

        # 定義畫布，大小足夠容納攝影機畫面和遊戲畫面
        canvas_width = 933
        canvas_height = 1080
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas.fill(244)  # 填充白色背景

        # 設定攝影機畫面的位置
        camera_x = 641
        camera_y = 150

        # 水平翻轉畫面
        flipped_frame = cv2.flip(frame, 1)

        # 定義裁切和調整大小的範圍
        crop_x_start, crop_x_end = 100, 500
        crop_y_start, crop_y_end = 50, 350
        resize_width, resize_height = mode_image_width, mode_image_height

        # 裁切和調整大小
        cropped_frame = flipped_frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        resized_frame = cv2.resize(cropped_frame, (resize_width, resize_height))

        # 將縮放後的攝影機畫面放到畫布上
        canvas[camera_y:camera_y + resize_height, camera_x:camera_x + resize_width] = resized_frame

        # 設定遊戲畫面的位置
        game_x = 116
        game_y = 300
        
        # 顯示分數畫面
        canvas[score_image_y:score_image_y + score_image_height, score_image_x:score_image_x + score_image_width] = Score_img
        score_text = f"{game.total_score}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = score_image_x + (score_image_width - text_size[0]) // 2
        text_y = score_image_y + (score_image_height + text_size[1]) // 2
        cv2.putText(canvas, score_text, (text_x, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # 監聽鍵盤事件，切換控制模式（Tab 鍵）
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        if key == 9 and current_time - last_mode_switch_time > 0.3:
            use_head_pose = not use_head_pose  # 切換控制模式
            last_mode_switch_time = current_time

        # 切換 AI 控制模式（按 'm' 鍵）
        if key == ord('m') and current_time - last_mode_switch_time > 0.3:
            use_ai_control = not use_ai_control
            last_mode_switch_time = current_time

        # 顯示當前控制模式
        if use_ai_control:
            canvas[mode_image_y:mode_image_y + mode_image_height, mode_image_x:mode_image_x + mode_image_width] = AI_mode_img
        elif use_head_pose:
            canvas[mode_image_y:mode_image_y + mode_image_height, mode_image_x:mode_image_x + mode_image_width] = Head_mode_img
        else:
            canvas[mode_image_y:mode_image_y + mode_image_height, mode_image_x:mode_image_x + mode_image_width] = WASD_mode_img

        # 處理頭部姿勢控制邏輯
        if use_head_pose and not use_ai_control:
            results = yolo_model(frame, verbose=False, device='cpu')  # 偵測人臉
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    face_img = frame[y1:y2, x1:x2]  # 擷取臉部圖像
                    if face_img.size == 0:
                        continue
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img = Image.fromarray(face_img)
                    face_img = transformations(face_img)
                    face_img = face_img.unsqueeze(0).to(device)

                    # 頭部姿勢預測
                    yaw, pitch, roll = pose_model(face_img)
                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)

                    # 預測頭部姿勢
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                    # 判斷頭部位置並執行相應操作
                    head_position = classify_head_pose(yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item())
                    if head_control.should_trigger_move(head_position):
                        if head_position == "Head Left":
                            game.step(0)  # 向左移動
                        elif head_position == "Head Right":
                            game.step(2)  # 向右移動
                        elif head_position == "Head Up":
                            game.step(1)  # 向上移動
                        elif head_position == "Head Down":
                            game.step(3)  # 向下移動

        # 處理 AI 控制邏輯
        elif use_ai_control:
            if use_ai_control and current_time - last_ai_move_time > ai_playing_speed:
                state = game.board
                action = predict_move_with_model(policy_model, state, device)
                game.step(action)
                last_ai_move_time = current_time

        # 處理 WASD 控制邏輯
        else:
            if current_time - last_key_time > debounce_time:
                if key in [ord('w'), ord('a'), ord('s'), ord('d')]:
                    if key == ord('w'):
                        game.step(1)  # 向上
                    elif key == ord('s'):
                        game.step(3)  # 向下
                    elif key == ord('a'):
                        game.step(0)  # 向左
                    elif key == ord('d'):
                        game.step(2)  # 向右
                    last_key_time = current_time

        # 繪製遊戲板畫面
        game_frame = np.zeros((700, 700, 3), dtype=np.uint8)
        game.draw_board(game_frame)
        resized_game_frame = cv2.resize(game_frame, (700, 700))
        canvas[game_board_y:game_board_y + 700, game_board_x:game_board_x + 700] = resized_game_frame

        # 顯示畫布（攝影機畫面在左邊，遊戲畫面在右邊）
        cv2.imshow("2048 and Head Pose Detection", canvas)

        # 檢查是否遊戲結束
        if game.game_over:
            overlay = np.full_like(canvas, 255)  # 透明覆蓋層
            alpha = 0.6  # 透明度
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

            text = "Nice Game!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 8)[0]
            text_x = (canvas_width - text_size[0]) // 2
            text_y = (canvas_height + text_size[1]) // 2 - 50
            cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

            sub_text = "Press any key to continue"
            sub_text_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            sub_text_x = (canvas_width - sub_text_size[0]) // 2
            sub_text_y = text_y + 50
            cv2.putText(canvas, sub_text, (sub_text_x, sub_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("2048 and Head Pose Detection", canvas)
            print("Game Over! Your score:", game.total_score)
            cv2.waitKey(0)
            game.reset()

        # 按 'q' 退出遊戲
        if key == ord('q'):
            break

    cap.release()  # 釋放攝影機資源
    cv2.destroyAllWindows()  # 關閉所有視窗

if __name__ == "__main__":
    main()
