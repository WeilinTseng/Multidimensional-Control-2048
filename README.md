# Multidimensional-Control-2048

## Overview
Multidimensional Control 2048 integrates head pose detection with AI-driven decision-making to provide an innovative approach to controlling the classic 2048 game. This project is designed to showcase the synergy between computer vision, machine learning, and gaming.

---

## Features
1. **2048 Game Core**:
   - Implements core game logic such as initialization, tile merging, movement, and game-over detection.
   - Supports reinforcement learning for automated gameplay optimization.

2. **Hopenet Training**:
   - Provides a framework for training the Hopenet model to estimate head pose (yaw, pitch, roll).
   - Includes data preprocessing, loss computation, and model evaluation.
   - 
3. **Head Pose Integration**:
   - Integrates head pose detection for multidimensional control of the 2048 game.
   - Enables gameplay using head movements via AI models.


---

## File Descriptions
### 1. **`2048_reinforcement.py`**
   - Implements the basic logic of the 2048 game.
   - Includes a deep reinforcement learning framework for training and automated gameplay.

### 2. **`trainHopenet.py`**
   - Trains the Hopenet model for head pose estimation.
   - Handles dataset loading, training, evaluation, and metric visualization.

### 3. **`head_pose_2048_integration.py`**
   - Integrates head pose detection with 2048 gameplay.
   - Supports control through head movements or automated AI-based decisions.

