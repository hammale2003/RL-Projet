# Double DQN (extension)

Documentation courte pour l’extension **Double DQN** sur le même benchmark `highway-v0` que le DQN vanilla, avec les mêmes hyperparamètres de base.

## Idée

Le DQN classique utilise une cible avec **max** sur le réseau cible, ce qui tend à **surestimer** les valeurs Q. Double DQN **découple** le choix de l’action au prochain état (réseau en ligne) et l’évaluation de cette action (réseau cible) :

\[
a^* = \arg\max_{a'} Q_\theta(s', a')
,\qquad
y = r + \gamma (1 - \text{done})\, Q_{\theta^-}(s', a^*)
\]

**Dans le code.** `double_dqn_agent.py` sous-classe le DQN et remplace uniquement le calcul de la cible dans `train_step`.

## Fichiers de résultats

Typiquement `results/double_dqn/seed_*/` (métriques, `eval_callback/`), et checkpoints sous `checkpoints/double_dqn/seed_*/` (`double_dqn_best.pt`, etc.).

## Référence

Van Hasselt et al., *Deep Reinforcement Learning with Double Q-learning*.
