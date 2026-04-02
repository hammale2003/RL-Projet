# RL Project – Partie 1 uniquement

Ce dépôt contient uniquement le code nécessaire pour la **partie 1 / core task** :
- un **DQN codé à la main**,
- un **modèle Stable-Baselines3** sur le même benchmark,
- l'entraînement, la sauvegarde des checkpoints,
- l'évaluation sur **50 épisodes**,
- les courbes,
- l'enregistrement d'un rollout vidéo.

## 1. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Structure

```text
rl_project_p1_final/
├── README.md
├── requirements.txt
├── checkpoints/
├── results/
├── videos/
└── src/
    ├── config_core.py
    ├── utils.py
    ├── models.py
    ├── replay_buffer.py
    ├── dqn_agent.py
    ├── train_dqn.py
    ├── train_sb3.py
    ├── evaluate.py
    ├── plot_results.py
    └── record_rollout.py
```

## 3. Entraîner le DQN maison

```bash
cd src
python train_dqn.py --seeds 0 1 2 --total-timesteps 50000 --eval-every 10000 --eval-episodes 10 --device cuda
```

## 4. Entraîner le modèle Stable-Baselines3

Ici j'ai choisi **SB3 DQN** pour garder une comparaison propre avec le DQN maison.

```bash
cd src
python train_sb3.py --seeds 0 1 2 --total-timesteps 50000 --eval-every 10000 --eval-episodes 10 --checkpoint-every 50000 --device cuda
```

## 5. Évaluation officielle sur 50 épisodes

### Éval DQN maison

```bash
cd src
python evaluate.py --model-type custom_dqn --model-template ../checkpoints/dqn/seed_0/dqn_best.pt --seeds 0 1 2 --num-episodes 50 --output-subdir evaluation_model_seed_0
python evaluate.py --model-type custom_dqn --model-template ../checkpoints/dqn/seed_1/dqn_best.pt --seeds 0 1 2 --num-episodes 50 --output-subdir evaluation_model_seed_1
python evaluate.py --model-type custom_dqn --model-template ../checkpoints/dqn/seed_2/dqn_best.pt --seeds 0 1 2 --num-episodes 50 --output-subdir evaluation_model_seed_2
```

### Éval SB3 DQN

```bash
cd src
python evaluate.py --model-type sb3_dqn --model-template ../checkpoints/sb3_dqn/seed_0/best_model/best_model.zip --seeds 0 1 2 --num-episodes 50 --output-subdir evaluation_model_seed_0
```

## 6. Courbes d'entraînement

```bash
cd src
python plot_results.py --seed 0 --model custom_dqn
python plot_results.py --seed 0 --model sb3_dqn
python plot_results.py --seed 0 --model both
```

Les figures seront sauvegardées dans `results/plots/`.

## 7. Enregistrer un rollout vidéo

### DQN maison

```bash
cd src
 python record_rollout.py --model-type custom_dqn --model-path ../checkpoints/dqn/seed_1/dqn_best.pt --seeds 0 1 2 --video-duration 80 --fps 5
```

### SB3 DQN

```bash
cd src
python record_rollout.py --model-type sb3_dqn --model-path ../checkpoints/sb3_dqn/seed_1/best_model/best_model.zip --seeds 0 1 2 --video-duration 80 --fps 5
```

Les vidéos seront dans `videos/`.
>>>>>>> 0ed9f23 (Initial commit)
