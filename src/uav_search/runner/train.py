from __future__ import annotations

import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from uav_search.algorithms.factory import make_algorithm
from uav_search.config import load_config
from uav_search.envs.paper_env import PaperUAVEnv
from uav_search.runtime import configure_runtime, resolve_device

from .logging import RunLogger
from .visualize import plot_training_curves, plot_trajectory
from .wandb_logger import WandbLogger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _obs_array(obs_dict: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    return np.stack([obs_dict[a] for a in agents], axis=0).astype(np.float32)


def deterministic_rollout(algo, cfg: dict[str, Any], seed: int) -> tuple[PaperUAVEnv, dict[str, float]]:
    env = PaperUAVEnv(cfg, seed=seed)
    obs_dict, _ = env.reset(seed=seed)
    obs = _obs_array(obs_dict, env.agents)
    returns = np.zeros(env.n_agents, dtype=np.float64)
    info: dict[str, Any] = {}
    collision_sum = obstacle_sum = boundary_sum = 0
    comm_rates: list[float] = []
    saturations: list[float] = []
    for _ in range(env.max_steps):
        action = algo.act(obs, explore=False)
        action_dict = {a: action[i] for i, a in enumerate(env.agents)}
        next_dict, reward_dict, _, truncated, info = env.step(action_dict)
        returns += np.asarray([reward_dict[a] for a in env.agents], dtype=np.float64)
        collision_sum += int(info["collisions"])
        obstacle_sum += int(info["obstacle_hits"])
        boundary_sum += int(info["boundary_hits"])
        comm_rates.append(float(info["mean_comm_rate_mbps"]))
        saturations.append(float(info["action_saturation"]))
        obs = _obs_array(next_dict, env.agents)
        if all(truncated.values()):
            break
    metrics = {
        "return_mean": float(returns.mean()),
        "return_sum": float(returns.sum()),
        "search_rate": float(info.get("search_rate", 0.0)),
        "targets_found": int(info.get("targets_found", 0)),
        "energy_j": float(env.total_energy_used_j),
        "mean_broken_link_s": float(info.get("mean_broken_link_s", 0.0)),
        "max_broken_link_s": float(info.get("max_broken_link_s", 0.0)),
        "collisions": collision_sum,
        "obstacle_hits": obstacle_sum,
        "boundary_hits": boundary_sum,
        "min_battery_pct": float(info.get("min_battery_pct", 100.0)),
        "mean_comm_rate_mbps": float(np.mean(comm_rates)) if comm_rates else 0.0,
        "action_saturation": float(np.mean(saturations)) if saturations else 0.0,
    }
    return env, metrics


def train_experiment(
    algorithm: str,
    scenario: str,
    episodes: int | None = None,
    steps: int | None = None,
    seed: int = 0,
    device: str = "auto",
    output_root: str | Path = "runs",
    wandb: bool = False,
    wandb_project: str = "uav-search-paper-baselines",
    run_name: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    deterministic: bool = False,
    amp_mode: str = "auto",
) -> Path:
    cfg = deepcopy(load_config(algorithm, scenario))
    cfg["runtime"]["seed"] = int(seed)
    if episodes is not None:
        cfg["runtime"]["episodes"] = int(episodes)
    if runtime_overrides:
        cfg["runtime"].update(runtime_overrides)
    if steps is not None:
        cfg["runtime"]["episode_steps_override"] = int(steps)
        cfg["paper"]["episode_steps"] = int(steps)
    profile = configure_runtime(resolve_device(device), deterministic=deterministic, amp_mode=amp_mode)
    device = str(profile.device)
    cfg["runtime"]["resolved_device"] = device
    cfg["runtime"]["deterministic"] = bool(deterministic)
    cfg["runtime"]["amp_mode"] = str(amp_mode)
    cfg["runtime"]["runtime_profile"] = profile.as_dict()
    set_global_seed(seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    name = run_name or stamp
    run_dir = Path(output_root) / algorithm.lower() / scenario / name
    logger = RunLogger(run_dir, low_window=int(cfg["runtime"]["low_episode_window"]))
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = PaperUAVEnv(cfg, seed=seed)
    algo = make_algorithm(algorithm, env, cfg, device=device, seed=seed)
    wb = WandbLogger(wandb, wandb_project, f"{algorithm}-{scenario}-{name}", cfg)
    rng = np.random.default_rng(seed + 12345)
    global_step = 0
    update_index = 0
    best_score = -float("inf")
    latest_update: dict[str, float] = {}
    current_context: dict[str, Any] = {"algorithm": algorithm, "scenario": scenario, "episode": 0, "step": 0}

    try:
        for episode in range(1, int(cfg["runtime"]["episodes"]) + 1):
            current_context["episode"] = episode
            obs_dict, _ = env.reset(seed=seed + episode - 1)
            obs = _obs_array(obs_dict, env.agents)
            ep_returns = np.zeros(env.n_agents, dtype=np.float64)
            collision_sum = obstacle_sum = boundary_sum = 0
            comm_rates: list[float] = []
            saturations: list[float] = []
            last_info: dict[str, Any] = {}

            for step in range(1, env.max_steps + 1):
                current_context["step"] = step
                if global_step < int(cfg["runtime"]["warmup_steps"]):
                    action = rng.uniform(-1.0, 1.0, size=(env.n_agents, env.action_dim)).astype(np.float32)
                else:
                    action = algo.act(obs, explore=True)
                action_dict = {a: action[i] for i, a in enumerate(env.agents)}
                next_dict, reward_dict, terminated, truncated, last_info = env.step(action_dict)
                next_obs = _obs_array(next_dict, env.agents)
                rewards = np.asarray([reward_dict[a] for a in env.agents], dtype=np.float32)
                dones = np.asarray(
                    [float(terminated[a] or truncated[a]) for a in env.agents], dtype=np.float32
                )
                algo.store(obs, action, rewards, next_obs, dones)
                ep_returns += rewards
                obs = next_obs
                global_step += 1
                collision_sum += int(last_info["collisions"])
                obstacle_sum += int(last_info["obstacle_hits"])
                boundary_sum += int(last_info["boundary_hits"])
                comm_rates.append(float(last_info["mean_comm_rate_mbps"]))
                saturations.append(float(last_info["action_saturation"]))

                if (
                    global_step >= int(cfg["runtime"]["update_after"])
                    and global_step % int(cfg["runtime"]["update_every"]) == 0
                ):
                    for _ in range(int(cfg["runtime"]["gradient_steps"])):
                        metrics = algo.update()
                        if metrics:
                            update_index += 1
                            latest_update = metrics
                            logger.log_update({"update": update_index, "global_step": global_step, **metrics})
                if all(truncated.values()) or all(terminated.values()):
                    break

            ep_metrics: dict[str, Any] = {
                "episode": episode,
                "global_step": global_step,
                "return_mean": float(ep_returns.mean()),
                "return_sum": float(ep_returns.sum()),
                "search_rate": float(last_info.get("search_rate", 0.0)),
                "targets_found": int(last_info.get("targets_found", 0)),
                "energy_j": float(env.total_energy_used_j),
                "mean_broken_link_s": float(last_info.get("mean_broken_link_s", 0.0)),
                "max_broken_link_s": float(last_info.get("max_broken_link_s", 0.0)),
                "collisions": collision_sum,
                "obstacle_hits": obstacle_sum,
                "boundary_hits": boundary_sum,
                "min_battery_pct": float(last_info.get("min_battery_pct", 100.0)),
                "action_saturation": float(np.mean(saturations)) if saturations else 0.0,
                "mean_comm_rate_mbps": float(np.mean(comm_rates)) if comm_rates else 0.0,
                "critic_loss": float(latest_update.get("critic_loss", 0.0)),
                "actor_loss": float(latest_update.get("actor_loss", 0.0)),
                "entropy": float(latest_update.get("entropy", 0.0)),
                "alpha": float(latest_update.get("alpha", 0.0)),
            }
            logger.log_episode(ep_metrics)
            wb.log({f"train/{k}": v for k, v in ep_metrics.items() if k != "episode"}, step=episode)
            score = ep_metrics["return_mean"] + 100.0 * ep_metrics["search_rate"]
            if cfg["runtime"].get("save_best", True) and score > best_score:
                best_score = score
                algo.save(logger.checkpoint_dir / "best.pt")

        algo.save(logger.checkpoint_dir / "final.pt")
        plot_training_curves(logger.episode_csv, logger.plots_dir)

        eval_env, eval_metrics = deterministic_rollout(algo, cfg, seed=seed + 100_000)
        rollout_dir = run_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            rollout_dir / "final_trajectory.npz",
            trajectory=np.asarray(eval_env.trajectory),
            targets=eval_env.targets,
            target_found=eval_env.target_found,
            obstacles=eval_env.obstacles,
            agent_names=np.asarray(eval_env.agents),
            agent_types=eval_env.agent_types,
        )
        plot_trajectory(
            np.asarray(eval_env.trajectory), eval_env.targets, eval_env.obstacles,
            eval_env.agents, eval_env.agent_types, logger.plots_dir / "trajectory.png", eval_env.area_size_m,
        )
        with (run_dir / "evaluation.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "algorithm": algorithm, "scenario": scenario, "episodes": int(cfg["runtime"]["episodes"]),
                "global_steps": global_step, "device": device, "best_score": best_score,
                "evaluation": eval_metrics,
            }, f, indent=2, ensure_ascii=False)
        return run_dir
    except BaseException as exc:
        logger.log_error(exc, current_context)
        try:
            algo.save(logger.checkpoint_dir / "crash.pt")
        except BaseException as save_exc:
            logger.log_error(save_exc, {**current_context, "during": "crash_checkpoint"})
        raise
    finally:
        wb.finish()
