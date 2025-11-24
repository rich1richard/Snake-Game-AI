from game_inputs.ai_agent import AIAgent
from snakegame import SnakeGame

# last model trained here
game = SnakeGame(bloc_size=10, width_bloc=40, height_blocs=40)

# test zone
# game = SnakeGame(bloc_size=5, width_bloc=100, height_blocs=100)

# agent = AIAgent(game, render=True)
agent = AIAgent(game, render=True, model_filename='model_1720984099_77.pth')

score, best_score = 0, 0

while True:
    agent.update()

    score = game.score()
    if score > best_score:
        best_score = score

    if agent.must_stop_game():
        break


print(f'GAME OVER... score: {score}, best score: {best_score}')
game.close()
