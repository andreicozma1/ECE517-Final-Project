def discount_rewards(rewards: list, gamma: float) -> list:
    cumul_reward = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r = (sum_r * gamma) + r
        cumul_reward.append(sum_r)
    return list(reversed(cumul_reward))
