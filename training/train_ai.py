"""
AI Training System - Complete Training Pipeline for Brawl Stars AI

Este script treina a IA usando múltiplos métodos:
1. Imitation Learning (aprende de replays/dados)
2. PPO Reinforcement Learning (treino principal)
3. Curriculum Learning (dificuldade progressiva)
4. Self-Play (opcional)

Uso:
    python training/train_ai.py                    # Treino completo
    python training/train_ai.py --method ppo      # Apenas PPO
    python training/train_ai.py --episodes 100    # 100 episódios
    python training/train_ai.py --eval-only       # Apenas avaliação
"""

import argparse
import asyncio
import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_training")


class AITrainingSystem:
    """
    Sistema completo de treino de IA para Brawl Stars.
    Integra PPO, Imitation Learning, Curriculum e Self-Play.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        self.rl_framework = None
        self.imitation_learning = None
        self.curriculum = None
        self.env = None

        self.training_history = []
        self.best_model = None
        self.best_win_rate = 0.0

        self._initialize_components()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "training": {
                "method": "combined",
                "total_episodes": 1000,
                "eval_every": 50,
                "save_every": 100,
                "verbose": True,
            },
            "imitation": {
                "enabled": True,
                "pretrain_epochs": 10,
                "min_demos": 100,
                "data_path": "pylaai_workspace/match_history.json",
            },
            "rl": {
                "enabled": True,
                "algorithm": "PPO",
                "total_timesteps": 50000,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "n_steps": 2048,
                "batch_size": 64,
            },
            "curriculum": {
                "enabled": True,
                "start_difficulty": 0.1,
                "max_difficulty": 1.0,
                "increase_rate": 0.05,
            },
            "evaluation": {
                "episodes": 20,
                "render": False,
            }
        }

    def _initialize_components(self):
        """Inicializa componentes de treino."""
        logger.info("=" * 60)
        logger.info("INICIALIZANDO SISTEMA DE TREINO")
        logger.info("=" * 60)

        try:
            from enterprise.learning.rl import RLFramework
            from enterprise.learning.imitation import ImitationLearning
            from enterprise.learning.curriculum import CurriculumLearning
            from enterprise.environments.real_env import RealBrawlStarsEnvironment

            logger.info("[1/4] Carregando RL Framework...")
            self.rl_framework = RLFramework(
                state_dim=128,
                action_dim=8,
                config=self.config.get("rl", {})
            )
            logger.info(f"      SB3 Available: {self.rl_framework.sb3_available}")

            logger.info("[2/4] Carregando Imitation Learning...")
            self.imitation_learning = ImitationLearning(state_dim=128, action_dim=8)

            logger.info("[3/4] Carregando Curriculum Learning...")
            self.curriculum = CurriculumLearning(task_fn=lambda p: {})

            logger.info("[4/4] Carregando Ambiente...")
            self.env = RealBrawlStarsEnvironment()

            logger.info("=" * 60)
            logger.info("COMPONENTES CARREGADOS COM SUCESSO!")
            logger.info("=" * 60)

        except ImportError as e:
            logger.error(f"Erro ao importar componentes: {e}")
            raise

    def load_demonstrations(self) -> int:
        """Carrega demonstrações de replays."""
        data_path = self.config.get("imitation", {}).get("data_path", "")

        if not os.path.exists(data_path):
            logger.warning(f"Ficheiro de replays não encontrado: {data_path}")
            logger.info("Gerando demonstrações sintéticas...")
            return self._generate_synthetic_demos()

        try:
            with open(data_path, 'r') as f:
                data = json.load(f)

            demos = data.get('matches', []) if isinstance(data, dict) else data
            logger.info(f"Carregados {len(demos)} replays")

            for match in demos:
                if isinstance(match, dict):
                    states = match.get('states', [])
                    actions = match.get('actions', [])

                    for i in range(min(len(states), len(actions))):
                        state = states[i] if i < len(states) else None
                        action = actions[i] if i < len(actions) else 0

                        if state is not None:
                            self.imitation_learning.add_demonstration(
                                state if isinstance(state, list) else [state],
                                action
                            )

            logger.info(f"Total demonstrações: {len(self.imitation_learning.demonstrations)}")
            return len(self.imitation_learning.demonstrations)

        except Exception as e:
            logger.error(f"Erro ao carregar replays: {e}")
            return self._generate_synthetic_demos()

    def _generate_synthetic_demos(self) -> int:
        """Gera demonstrações sintéticas para bootstrap."""
        logger.info("Gerando demonstrações sintéticas...")

        import numpy as np

        for ep in range(50):
            state = np.random.randn(128).astype(np.float32)

            enemy_pos = np.random.choice(8)
            action = int(enemy_pos)

            self.imitation_learning.add_demonstration(state, action)

        logger.info(f"Geradas {len(self.imitation_learning.demonstrations)} demonstrações sintéticas")
        return len(self.imitation_learning.demonstrations)

    def pretrain_imitation(self) -> Dict[str, float]:
        """Pré-treino com Imitation Learning."""
        if not self.config.get("imitation", {}).get("enabled", True):
            logger.info("Imitation Learning desativado")
            return {"status": "disabled"}

        min_demos = self.config.get("imitation", {}).get("min_demos", 100)
        epochs = self.config.get("imitation", {}).get("pretrain_epochs", 10)

        demos = len(self.imitation_learning.demonstrations)
        if demos < min_demos:
            logger.warning(f"Demonstrações insuficientes: {demos} < {min_demos}")
            logger.info(f"Gerando mais demonstrações...")
            for _ in range(min_demos - demos):
                import numpy as np
                state = np.random.randn(128).astype(np.float32)
                action = np.random.randint(0, 8)
                self.imitation_learning.add_demonstration(state, action)

        logger.info("=" * 60)
        logger.info("IMITATION LEARNING - PRÉ-TREINO")
        logger.info("=" * 60)

        try:
            result = self.imitation_learning.pretrain(epochs)
            logger.info(f"Pré-treino completo!")
            logger.info(f"Loss médio: {result.get('avg_loss', 0):.4f}")
            logger.info(f"Épocas: {result.get('epochs', 0)}")
        except Exception as e:
            logger.warning(f"Imitation Learning pretrain failed: {e}")
            logger.info("Skipping imitation learning, continuing with RL training...")
            result = {"status": "skipped", "reason": str(e)}

        return result

    def train_rl(self, total_timesteps: int = None) -> Dict[str, Any]:
        """Treino com PPO."""
        if not self.config.get("rl", {}).get("enabled", True):
            logger.info("RL desativado")
            return {"status": "disabled"}

        if total_timesteps is None:
            total_timesteps = self.config.get("rl", {}).get("total_timesteps", 50000)

        logger.info("=" * 60)
        logger.info("PPO REINFORCEMENT LEARNING")
        logger.info("=" * 60)
        logger.info(f"Timesteps: {total_timesteps:,}")
        logger.info(f"Algorithm: {self.config.get('rl', {}).get('algorithm', 'PPO')}")

        if self.rl_framework.sb3_available:
            result = self._train_with_sb3(total_timesteps)
        else:
            result = self._train_mock(total_timesteps)

        return result

    def _train_with_sb3(self, total_timesteps: int) -> Dict[str, Any]:
        """Treino com Stable-Baselines3 real."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            def make_env():
                def _init():
                    from enterprise.environments.real_env import RealBrawlStarsEnvironment
                    return RealBrawlStarsEnvironment()
                return _init

            vec_env = DummyVecEnv([make_env()])

            rl_config = self.config.get("rl", {})

            model = PPO(
                "CnnPolicy",
                vec_env,
                learning_rate=rl_config.get("learning_rate", 3e-4),
                n_steps=rl_config.get("n_steps", 2048),
                batch_size=rl_config.get("batch_size", 64),
                gamma=rl_config.get("gamma", 0.99),
                verbose=1 if rl_config.get("verbose", True) else 0,
            )

            logger.info("A treinar modelo PPO (Stable-Baselines3)...")
            start_time = time.time()

            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                reset_num_timesteps=False
            )

            elapsed = time.time() - start_time

            model_path = f"models/ppo_brawlstars_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save(model_path)
            logger.info(f"Modelo guardado em: {model_path}")

            self.rl_framework.model = model

            return {
                "status": "trained",
                "method": "stable_baselines3",
                "timesteps": total_timesteps,
                "time_elapsed": elapsed,
                "model_path": model_path
            }

        except Exception as e:
            logger.error(f"Erro no treino SB3: {e}")
            return self._train_mock(total_timesteps)

    async def _train_mock(self, total_timesteps: int) -> Dict[str, Any]:
        """Treino mock (sem SB3)."""
        logger.info("A usar implementation mock (sem SB3)...")

        start_time = time.time()
        episodes = 0

        for step in range(min(total_timesteps // 1000, 100)):
            await asyncio.sleep(0.01)

            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.rl_framework.select_action(
                    obs.flatten().astype(float)
                )
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                self.rl_framework.store(
                    obs.flatten().astype(float),
                    action,
                    reward,
                    obs.flatten().astype(float),
                    done
                )

            episodes += 1

            if step % 10 == 0:
                loss = self.rl_framework.train(batch_size=64).get("loss", 0)
                logger.info(f"Step {step}/100 | Reward: {episode_reward:.2f} | Loss: {loss:.4f}")

        elapsed = time.time() - start_time

        self.rl_framework.q_network = {"trained": True, "mock": True}

        return {
            "status": "trained",
            "method": "mock",
            "timesteps": total_timesteps,
            "episodes": episodes,
            "time_elapsed": elapsed
        }

    def evaluate(self, episodes: int = None) -> Dict[str, Any]:
        """Avalia o modelo treinado."""
        if episodes is None:
            episodes = self.config.get("evaluation", {}).get("episodes", 20)

        logger.info("=" * 60)
        logger.info(f"AVALIAÇÃO - {episodes} EPISÓDIOS")
        logger.info("=" * 60)

        wins = 0
        total_rewards = []
        total_steps = []

        for ep in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < self.env.max_steps:
                action = self.rl_framework.select_action(
                    obs.flatten().astype(float),
                    training=False
                )
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                if done:
                    break

            win = self.env.state["player_health"] > 0 or any(
                e["health"] <= 0 for e in self.env.state["enemies"]
            )
            if win:
                wins += 1

            total_rewards.append(episode_reward)
            total_steps.append(steps)

            logger.info(f"Episódio {ep+1}/{episodes} | Steps: {steps} | Reward: {episode_reward:.2f} | {'WIN' if win else 'LOSS'}")

        win_rate = wins / episodes
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_steps = sum(total_steps) / len(total_steps)

        logger.info("=" * 60)
        logger.info("RESULTADOS DA AVALIAÇÃO")
        logger.info("=" * 60)
        logger.info(f"Win Rate: {win_rate*100:.1f}%")
        logger.info(f"Avg Reward: {avg_reward:.2f}")
        logger.info(f"Avg Steps: {avg_steps:.1f}")

        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            logger.info(f"NOVO MELHOR MODELO! Win Rate: {win_rate*100:.1f}%")

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "wins": wins,
            "episodes": episodes,
            "is_best": win_rate >= self.best_win_rate
        }

    def run_full_training(self) -> Dict[str, Any]:
        """Executa pipeline completo de treino."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETO DE TREINO")
        logger.info("=" * 60)

        total_episodes = self.config.get("training", {}).get("total_episodes", 1000)
        eval_every = self.config.get("training", {}).get("eval_every", 50)
        save_every = self.config.get("training", {}).get("save_every", 100)

        results = {}

        logger.info("\n[FASE 1] Imitation Learning Bootstrap")
        logger.info("-" * 40)
        imitation_result = self.pretrain_imitation()
        results["imitation"] = imitation_result

        logger.info("\n[FASE 2] PPO Reinforcement Learning")
        logger.info("-" * 40)
        timesteps = self.config.get("rl", {}).get("total_timesteps", 50000)
        rl_result = asyncio.get_event_loop().run_until_complete(
            self.train_rl(timesteps)
        )
        results["rl"] = rl_result

        logger.info("\n[FASE 3] Avaliação Intermédia")
        logger.info("-" * 40)
        eval_result = self.evaluate(episodes=20)
        results["evaluation_initial"] = eval_result

        logger.info("\n[FASE 4] Curriculum Learning")
        logger.info("-" * 40)
        curriculum_result = self._run_curriculum()
        results["curriculum"] = curriculum_result

        logger.info("\n" + "=" * 60)
        logger.info("TREINO COMPLETO TERMINADO!")
        logger.info("=" * 60)
        logger.info(f"Melhor Win Rate: {self.best_win_rate*100:.1f}%")

        return results

    def _run_curriculum(self) -> Dict[str, Any]:
        """Executa curriculum learning."""
        if not self.config.get("curriculum", {}).get("enabled", True):
            return {"status": "disabled"}

        logger.info("Executing Curriculum Learning...")

        start_diff = self.config.get("curriculum", {}).get("start_difficulty", 0.1)
        max_diff = self.config.get("curriculum", {}).get("max_difficulty", 1.0)
        increase = self.config.get("curriculum", {}).get("increase_rate", 0.05)

        difficulty = start_diff
        stages_completed = 0

        while difficulty <= max_diff:
            logger.info(f"  Difficulty: {difficulty:.2f}")

            result = asyncio.get_event_loop().run_until_complete(
                self.train_rl(total_timesteps=5000)
            )

            eval_result = self.evaluate(episodes=10)

            if eval_result["win_rate"] > 0.5:
                difficulty += increase
                stages_completed += 1
                logger.info(f"  Passed! Advancing to difficulty {difficulty:.2f}")
            else:
                logger.info(f"  Needs more training at this level")

        return {
            "stages_completed": stages_completed,
            "final_difficulty": difficulty
        }

    def save_training_history(self, path: str = "models/training_history.json"):
        """Guarda histórico de treino."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        history = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "history": self.training_history,
            "best_win_rate": self.best_win_rate,
        }

        with open(path, 'w') as f:
            json.dump(history, f, indent=2, default=str)

        logger.info(f"Histórico guardado em: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="AI Training System for Brawl Stars")
    parser.add_argument("--method", type=str, default="combined",
                       choices=["combined", "ppo", "imitation", "curriculum"],
                       help="Training method")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Total episodes")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="RL timesteps")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path")
    return parser.parse_args()


def main():
    args = parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    if args.method != "combined":
        config["training"] = {"method": args.method}
    if args.episodes:
        config.setdefault("training", {})["total_episodes"] = args.episodes
    if args.timesteps:
        config.setdefault("rl", {})["total_timesteps"] = args.timesteps

    trainer = AITrainingSystem(config)

    if args.eval_only:
        logger.info("MODO: Apenas Avaliação")
        result = trainer.evaluate(episodes=20)
        print(json.dumps(result, indent=2))
    else:
        logger.info("MODO: Treino Completo")
        results = trainer.run_full_training()
        print(json.dumps(results, indent=2, default=str))

        trainer.save_training_history()


if __name__ == "__main__":
    main()
