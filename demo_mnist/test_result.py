from matplotlib import pyplot as plt
import torch as tc
device = tc.device('cuda:0')


def test_policy(env, action_labels, T, agent=None, fixed_action=None, randomize=False):
    done = False
    fig = plt.figure(figsize=(2, 2))
    plt.title('original')
    plt.imshow(env._b.cpu())
    plt.axis('off')
    plt.show()
    
    fig = plt.figure(figsize=(20, 2))
    plt.subplot(1, T + 1, 1)
    plt.axis('off')
    plt.imshow(env._state.now.cpu())
    print(T, '回での', action_labels[env.decay_kernel_index], 'での復元')
    took_actions = []
    reward_sum = 0
    while not done:
        if fixed_action is not None:
            a = fixed_action
        elif randomize:
            a = tc.randint(4, ()).to(device)
        else:
            a = agent.policy(env.state)
        took_actions.append(action_labels[a])
        next_state, reward, done = env.step(a)
        plt.subplot(1, T + 1, env._step_count + 1)
        plt.axis('off')
        plt.imshow(env._state.now.cpu())
        reward_sum += reward
        print(f'loss of step({env._step_count})', tc.sum((env._b - env._state.now)**2).item(), end='//')
    print()
    print('reward_sum: ', reward_sum.item())
    plt.show()
    print(took_actions)