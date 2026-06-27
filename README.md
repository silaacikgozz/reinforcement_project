# IE 306 Term Project — RL for City-Scale Drone Delivery

## Setup (one command)
```bash
pip install -r requirements.txt
```
This installs the course simulator (lives in `simulator/`, the `drone_dispatch_env`
package shipped unmodified inside this repo — folder renamed from the original zip
to `simulator/` to avoid a Python namespace-package collision with the inner
package of the same name) plus everything our learned methods need (torch, etc.).

## Sanity check
```bash
python -m pytest simulator/tests -q
bash simulator/reproduce.sh simulator/configs/eval_standard.yaml "0,1,2" greedy_nearest
```

## Repo layout
```
simulator/             <- instructor-provided drone_dispatch_env package (untouched)
code/
  common/              <- shared helpers (obs encoding, env wrappers, seeding) used by everyone
  role_a_dqn/           <- Role A: DQN -> Double DQN -> Dueling DQN  (owner: Kerem Ayyılmaz)
  role_b_policy/         <- Role B: REINFORCE/GAE -> A2C, DDPG       (owner: Nisa Üstünakın)
  role_c_planning/       <- Role C: Dyna-Q                          (owner: Sıla Açıkgöz)
  offline_rl/            <- Joint: naive-DQN-offline failure -> CQL fix
  multi_agent/           <- Joint: IDQN (parameter-shared) decentralized dispatcher
configs/                 <- one yaml per experiment (hyperparameters, no magic numbers in code)
weights/                 <- trained model checkpoints
logs/                    <- raw CSV logs behind every learning curve
run_all.py               <- loads saved weights, prints the baseline-comparison table
```

## Run a specific method
See each role's subfolder README (added as that method is implemented) for its exact
single-command invocation and config file.
