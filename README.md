# Snake Game AI

A reinforcement learning agent that plays the classic Snake game with real-time rendering and score tracking.

## What is this project?

This project implements a Snake game with an AI agent trained using reinforcement learning. The AI has been trained to make optimal decisions about which direction to move the snake to maximize its score while avoiding collisions with walls or its own body.

The system uses a pre-trained model (`model_1720984099_77.pth`) that has been trained to play the game efficiently.

## Key Features

* 🎮 Real-time Snake game with Pygame rendering
* 💡 Reinforcement learning agent with defined reward structure
* 📊 Score tracking and best score monitoring
* 🧠 Pre-trained model for immediate gameplay
* 🔍 Configurable game dimensions (block size, width, height)
* 📦 Modular architecture with clear separation of concerns

## How it works

The AI agent uses a reinforcement learning approach with the following reward structure:

| Action | Reward |
|--------|--------|
| Eats food | +10 |
| Normal movement | -0.01 |
| Game over (collision) | -10 |

The agent learns by:

1. Observing the game state
2. Choosing a direction to move
3. Receiving rewards based on the outcome
4. Updating its strategy to maximize future rewards

## Getting Started

### Installation

1. Install Python 3.7+ (recommended)
2. Install Pygame:

```bash
pip install pygame
```

### Running the Game

Simply run the main application:

```bash
python main.py
```

This will launch the Snake game with the pre-trained AI agent. The game will automatically:

* Render the snake on screen
* Track score and best score
* Play with the pre-trained model

## Model Information

The current model used is: `model_1720984099_77.pth`

This model was trained to achieve a best score of `77` in the game environment with the following parameters:

* Block size: 10
* Game width: 40
* Game height: 40

## Customization

You can customize the game parameters in the `main.py` file:

```python
game = SnakeGame(bloc_size=10, width_bloc=40, height_blocs=40)
```

Change the values to adjust the game size and difficulty.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is a research prototype for demonstrating reinforcement learning in a simple game environment. The model and training parameters are specific to this implementation and may not work with other Snake game implementations.

