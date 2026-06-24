import yaml
import gymnasium as gym
import drone_dispatch_env
import numpy as np
import pandas as pd
import torch
import os
import sys

# Klasör yollarını Python'a tanıtıyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from dqn_agent import DQNAgent
from dynaq_agent import DynaQAgent
from a2c_agent import A2CAgent

def get_flat_dim(obs):
    """Gözlem sözlüğünün boyutunu dinamik ve hatasız hesaplar."""
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

def train_agent(agent_type="dqn"):
    config_path = f"configs/{agent_type}.yaml"
    if not os.path.exists(config_path):
        print(f"HATA: {config_path} bulunamadı!")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    seeds = [0, 1, 2] # Proje isterindeki 3 zorunlu seed
    
    for seed in seeds:
        print(f"\n--- {agent_type.upper()} SEED {seed} EĞİTİMİ BAŞLADI ---")
        env = gym.make("DroneDispatch-v0")
        obs, _ = env.reset(seed=seed)
        
        state_dim = get_flat_dim(obs)
        action_dim = len(obs["action_mask"])
        
        # İstek yapılan ajana göre nesne oluşturuluyor
        if agent_type == "dqn":
            agent = DQNAgent(state_dim, action_dim, config)
        elif agent_type == "dynaq":
            agent = DynaQAgent(state_dim, action_dim, config)
        elif agent_type == "a2c":  
            agent = A2CAgent(state_dim, action_dim, config)
            
        global_step = 0
        steps_logged = []
        rewards_logged = []
        max_episodes = config.get("episodes", 200)
        
        for episode in range(max_episodes):
            obs, info = env.reset(seed=seed + episode)
            ep_reward = 0
            done = False
            
            while not done:
                # A2C ise çoklu dönüş alıyoruz, diğerleri ise sadece tek aksiyon alır
                if agent_type == "a2c":
                    action, log_prob, value = agent.act(obs)
                else:
                    action = agent.act(obs)
                    
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Ajan tiplerine göre öğrenme adımları tetikleniyor
                if agent_type == "dqn":
                    agent.store(obs, action, reward, next_obs, done)
                    agent.train_step()
                    if global_step % config.get("target_update_freq", 500) == 0 and agent.use_target:
                        agent.target_net.load_state_dict(agent.q_net.state_dict())
                elif agent_type == "dynaq":
                    agent.store_and_train(obs, action, reward, next_obs, done)
                elif agent_type == "a2c":  # <-- ADIM ADIM HAFIZAYA KAYDETME
                    agent.store(log_prob, value, reward, done)
                
                obs = next_obs
                ep_reward += reward
                global_step += 1
                
            # --- WHILE DÖNGÜSÜ BİTTİ (EPISODE SONU) ---
            # A2C bölüm sonunda toplu güncelleme yapar (On-Policy)
            if agent_type == "a2c":
                agent.update()
            
            if episode % 10 == 0 or episode == max_episodes - 1:
                # A2C'de epsilon olmadığı için print hatası vermemesi için kontrol ekledik
                eps_val = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
                print(f"Ep: {episode}/{max_episodes} | Step: {global_step} | Skor: {ep_reward:.2f} | Eps: {eps_val:.3f}")
                
            steps_logged.append(global_step)
            rewards_logged.append(ep_reward)
            
        # Logları ve ağırlıkları kaydetme
        df = pd.DataFrame({"step": steps_logged, "reward": rewards_logged})
        df.to_csv(f"logs/{agent_type}_seed{seed}.csv", index=False)
        
        if agent_type == "dqn":
            torch.save(agent.q_net.state_dict(), f"weights/dqn_seed{seed}.pt")
        elif agent_type == "dynaq":
            np.save(f"weights/dynaq_seed{seed}.npy", agent.q_table)
        elif agent_type == "a2c":  # <-- NİSA'NIN MODEL KAYDETME SATIRI
            torch.save(agent.network.state_dict(), f"weights/a2c_seed{seed}.pt")
            
        print(f"--- {agent_type.upper()} SEED {seed} KAYDEDİLDİ ---")

if __name__ == "__main__":
    # Eğer terminalden argüman verilmezse varsayılan olarak dynaq eğitir
    target = sys.argv[1] if len(sys.argv) > 1 else "dynaq"
    train_agent(target)