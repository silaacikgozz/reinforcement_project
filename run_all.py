import os
import sys
import yaml
import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, GreedyNearest, RandomPolicy
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "code")
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

from dqn_agent import DQNAgent
from dynaq_agent import DynaQAgent
from a2c_agent import A2CAgent  

class A2CEvalWrapper:
    """Simülatörün test esnasında sadece tek bir int aksiyon alması için sarmalayıcı sınıf."""
    def __init__(self, agent):
        self.agent = agent
    def act(self, obs):
        return self.agent.act(obs)[0]

def get_flat_dim(obs):
    flat_list = []
    for k, v in obs.items():
        if k != "action_mask":
            if isinstance(v, np.ndarray): flat_list.extend(v.flatten())
            elif isinstance(v, (int, float)): flat_list.append(v)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, np.ndarray): flat_list.extend(sv.flatten())
                    elif isinstance(sv, (int, float)): flat_list.append(sv)
    return len(flat_list)

def extract_mean_score(metrics_output):
    if isinstance(metrics_output, (int, float)): return float(metrics_output)
    if isinstance(metrics_output, dict):
        if "mean" in metrics_output:
            val = metrics_output["mean"]
            if isinstance(val, dict): return float(val.get("cost_per_order", list(val.values())[0]))
            return float(val)
        for k, v in metrics_output.items():
            if isinstance(v, (int, float)): return float(v)
    return 999.0

def evaluate_our_policy():
    env_config = Config()
    seeds = [0, 1, 2]
    
    print("Sistem baselineları test ediliyor...")
    rand_metrics = evaluate(RandomPolicy(env_config), env_config, seeds=seeds)
    greedy_metrics = evaluate(GreedyNearest(env_config), env_config, seeds=seeds)
    
    env = gym.make("DroneDispatch-v0")
    obs, _ = env.reset(seed=0)
    state_dim = get_flat_dim(obs)
    action_dim = len(obs["action_mask"])
                    
    # --- ROL A YÜKLEME ---
    with open("configs/dqn.yaml", "r") as f:
        config_dqn = yaml.safe_load(f)
    
    agent_dqn = DQNAgent(state_dim, action_dim, config_dqn)
    config_double = config_dqn.copy()
    config_double["use_double"] = True
    agent_double = DQNAgent(state_dim, action_dim, config_double)
    agent_dueling = DQNAgent(state_dim, action_dim, config_dqn)
    
    dqn_weight = os.path.join(CURRENT_DIR, "weights", "dqn_seed0.pt")
    if os.path.exists(dqn_weight):
        agent_dqn.q_net.load_state_dict(torch.load(dqn_weight, map_location=agent_dqn.device))
        agent_double.q_net.load_state_dict(torch.load(dqn_weight, map_location=agent_double.device))
        agent_dueling.q_net.load_state_dict(torch.load(dqn_weight, map_location=agent_dueling.device))
    
    agent_dqn.epsilon = 0.0
    agent_double.epsilon = 0.0
    agent_dueling.epsilon = 0.0

    # --- ROL C YÜKLEME ---
    with open("configs/dynaq.yaml", "r") as f:
        config_c = yaml.safe_load(f)
    agent_dynaq = DynaQAgent(state_dim, action_dim, config_c)
    
    dynaq_weight = os.path.join(CURRENT_DIR, "weights", "dynaq_seed0.npy")
    if os.path.exists(dynaq_weight):
        agent_dynaq.q_table = np.load(dynaq_weight, allow_pickle=True).item()
    agent_dynaq.epsilon = 0.0

    # --- ROL B YÜKLEME (NİSA) ---
    config_a2c = {}
    if os.path.exists("configs/a2c.yaml"):
        with open("configs/a2c.yaml", "r") as f:
            config_a2c = yaml.safe_load(f)
    agent_a2c = A2CAgent(state_dim, action_dim, config_a2c)
    
    a2c_weight = os.path.join(CURRENT_DIR, "weights", "a2c_seed0.pt")
    if os.path.exists(a2c_weight):
        agent_a2c.network.load_state_dict(torch.load(a2c_weight, map_location=agent_a2c.device))
    
    agent_a2c_eval = A2CEvalWrapper(agent_a2c)

    print("Bütün modeller test ediliyor (Bu işlem biraz sürebilir)...")
    metrics_dqn = evaluate(agent_dqn, env_config, seeds=seeds)
    metrics_double = evaluate(agent_double, env_config, seeds=seeds)
    metrics_dueling = evaluate(agent_dueling, env_config, seeds=seeds)
    metrics_dynaq = evaluate(agent_dynaq, env_config, seeds=seeds)
    metrics_a2c = evaluate(agent_a2c_eval, env_config, seeds=seeds)  
    
    # Skorları Çıkarma
    r_score = extract_mean_score(rand_metrics)
    g_score = extract_mean_score(greedy_metrics)
    score_dqn = extract_mean_score(metrics_dqn)
    score_double = extract_mean_score(metrics_double)
    score_dueling = extract_mean_score(metrics_dueling)
    score_dynaq = extract_mean_score(metrics_dynaq)
    score_a2c = extract_mean_score(metrics_a2c)  
    
    # Nihai Karşılaştırma Tablosu
    print("\n" + "="*55)
    print(f"{'Method':<25} | {'cost/order':<25}")
    print("="*55)
    print(f"{'random':<25} | {r_score:<25.4f}")
    print(f"{'greedy_nearest':<25} | {g_score:<25.4f}")
    print(f"{'DQN (dqn)':<25} | {score_dqn:<25.4f}")
    print(f"{'DQN (double)':<25} | {score_double:<25.4f}")
    print(f"{'DQN (dueling)':<25} | {score_dueling:<25.4f}")
    print(f"{'Dyna-Q':<25} | {score_dynaq:<25.4f}")
    print(f"{'A2C (Nisa)':<25} | {score_a2c:<25.4f}")  
    print("="*55)

if __name__ == "__main__":
    evaluate_our_policy()