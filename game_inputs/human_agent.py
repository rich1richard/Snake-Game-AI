from game_inputs import BaseAgent, Direction
import pygame


class HumanAgent(BaseAgent):
    def update(self):
        game = self.game()

        game.render()
        if game.must_quit():
            self.stop_game()

        direction = self._direction
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != Direction.DOWN:
                    direction = Direction.UP
                elif event.key == pygame.K_DOWN and direction != Direction.UP:
                    direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and direction != Direction.RIGHT:
                    direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and direction != Direction.LEFT:
                    direction = Direction.RIGHT

        if not self.must_stop_game():
            self.set_direction(direction)
            game_over, _ = game.step(direction)

            if game_over:
                self.stop_game()
