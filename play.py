from snake2 import Snake
env = Snake(5)
env.reset()
env.render()
total_reward = 0
print('Play snake with wasd:')
while True:
    action = input()
    correct_input = True
    if action == 'w':
        action_int = 0
    elif action == 's':
        action_int = 1
    elif action == 'a':
        action_int = 2
    elif action == 'd':
        action_int = 3
    else:
        action_int = -1
        print('Wrong input!')
        correct_input = False
    if (correct_input):
        observation, reward, terminated, info = env.step(action_int)
        env.render()
        total_reward += reward
        print('Reward: ', total_reward)
    if (terminated):
        break
print('Terminated')

