from __future__ import annotations

import json
import random
import time
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

from .diagnostics import diagnose_episode
from .logging import RunLogger
from .progress import format_episode_progress, format_startup_summary, should_report_episode
from .visualize import plot_simulation_scenario, plot_training_curves, plot_trajectory
from .wandb_logger import WandbLogger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _obs_array(obs_dict: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    return np.stack([obs_dict[a] for a in agents], axis=0).astype(np.float32)


def _replay_terminal_mask(
    terminated: dict[str, bool],
    agents: list[str],
) -> np.ndarray:
    """Return replay bootstrap stops: only true MDP termination cuts the target.

    Truncation still ends rollout collection, but a time-limit transition keeps
    bootstrapping from the next observation according to Gymnasium's step API.
    """
    return np.asarray([float(terminated[a]) for a in agents], dtype=np.float32)


def _sensing_metrics_from_info(info: dict[str, Any]) -> dict[str, float | int]:
    """Extract episode-level 3D sensing diagnostics with legacy-safe defaults."""
    return {
        "mean_altitude_m": float(info.get("mean_altitude_m", 0.0)),
        "min_altitude_m": float(info.get("min_altitude_m", 0.0)),
        "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
        "mean_belief_entropy": float(info.get("mean_belief_entropy", 0.0)),
        "mean_target_posterior": float(info.get("mean_target_posterior", 0.0)),
        "scanned_cells_total": int(info.get("scanned_cells_total", 0)),
        "positive_sensor_observations_total": int(info.get("positive_sensor_observations_total", 0)),
        "information_gain_total": float(info.get("information_gain_total", 0.0)),
        "targets_confirmed_total": int(info.get("targets_confirmed_total", 0)),
        "false_confirmations_total": int(info.get("false_confirmations_total", 0)),
        "confirmed_cells_total": int(info.get("confirmed_cells_total", 0)),
    }


def deterministic_rollout(algo, cfg: dict[str, Any], seed: int) -> tuple[PaperUAVEnv, dict[str, float]]:
    env = PaperUAVEnv(cfg, seed=seed)
    obs_dict, _ = env.reset(seed=seed)
    obs = _obs_array(obs_dict, env.agents)
    returns = np.zeros(env.n_agents, dtype=np.float64)
    info: dict[str, Any] = {}
    comm_rates: list[float] = []
    saturations: list[float] = []
    for _ in range(env.max_steps):
        action = algo.act(obs, explore=False)
        action_dict = {a: action[i] for i, a in enumerate(env.agents)}
        next_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
        returns += np.asarray([reward_dict[a] for a in env.agents], dtype=np.float64)
        comm_rates.append(float(info["mean_comm_rate_mbps"]))
        saturations.append(float(info["action_saturation"]))
        obs = _obs_array(next_dict, env.agents)
        if all(truncated.values()) or all(terminated.values()):
            break
    metrics = {
        "return_mean": float(returns.mean()),
        "return_sum": float(returns.sum()),
        "search_rate": float(info.get("search_rate", 0.0)),
        "targets_found": int(info.get("targets_found", 0)),
        "energy_j": float(env.total_energy_used_j),
        "energy_consumption_pct": float(info.get("energy_consumption_pct", 0.0)),
        "mean_broken_link_s": float(info.get("mean_broken_link_s", 0.0)),
        "max_broken_link_s": float(info.get("max_broken_link_s", 0.0)),
        "safety_distance_violation_uavs": int(info.get("safety_distance_violation_uavs", 0)),
        "obstacle_hit_uavs": int(info.get("obstacle_hit_uavs", 0)),
        "boundary_hit_uavs": int(info.get("boundary_hit_uavs", 0)),
        "broken_link_uavs": int(info.get("broken_link_uavs", 0)),
        "depleted_uavs": int(info.get("depleted_uavs", 0)),
        "avg_battery_pct": float(info.get("avg_battery_pct", 100.0)),
        "mean_comm_rate_mbps": float(np.mean(comm_rates)) if comm_rates else 0.0,
        "action_saturation": float(np.mean(saturations)) if saturations else 0.0,
        "mission_delivery_rate": float(info.get("mission_delivery_rate", 0.0)),
        "reports_delivered": int(info.get("reports_delivered", 0)),
        "expired_reports": int(info.get("expired_reports", 0)),
        "queue_bytes_total": int(info.get("queue_bytes_total", 0)),
        "network_byte_pdr": float(info.get("network_byte_pdr", 0.0)),
        "network_throughput_bps": float(info.get("network_throughput_bps", 0.0)),
        "direct_gcs_uavs": int(info.get("direct_gcs_uavs", 0)),
        "multihop_gcs_uavs": int(info.get("multihop_gcs_uavs", 0)),
        "disconnected_gcs_uavs": int(info.get("disconnected_gcs_uavs", 0)),
        "mean_gcs_hops": float(info.get("mean_gcs_hops", 0.0)),
        "battery_capacity_j": float(info.get("battery_capacity_j", 0.0)),
    }
    metrics.update(_sensing_metrics_from_info(info))
    return env, metrics


def train_experiment(
    algorithm: str,
    scenario: str,
    episodes: int | None = None,
    steps: int | None = None,
    seed: int = 44,
    device: str = "auto",
    output_root: str | Path = "runs",
    wandb: bool | None = None,
    wandb_api_key: str | None = None,
    run_name: str | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    deterministic: bool = True,
    amp_mode: str = "auto",
    wandb_mode: str = "auto",
    progress_every: int = 100,
) -> Path:
    if int(progress_every) < 1:
        raise ValueError("progress_every must be >= 1")

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
    cfg["runtime"]["progress_every"] = int(progress_every)
    set_global_seed(seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    name = run_name or stamp
    run_dir = Path(output_root) / algorithm.lower() / scenario / name
    logger = RunLogger(run_dir, low_window=int(cfg["runtime"]["low_episode_window"]))
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = PaperUAVEnv(cfg, seed=seed)
    algo = make_algorithm(algorithm, env, cfg, device=device, seed=seed)
    wb = WandbLogger(
        run_name=f"{algorithm}-{scenario}-{name}",
        config=cfg,
        run_dir=run_dir,
        api_key=wandb_api_key,
        mode=wandb_mode,
        enabled=wandb,
    )

    total_episodes = int(cfg["runtime"]["episodes"])
    print(
        format_startup_summary(
            algorithm=algorithm,
            scenario=scenario,
            episodes=total_episodes,
            steps=int(env.max_steps),
            device=device,
            device_name=profile.device_name,
            amp_enabled=profile.amp_enabled,
            deterministic=profile.deterministic,
            tracking_status=wb.status,
            project_url=wb.project_url,
            run_url=wb.run_url,
            run_dir=str(run_dir),
        ),
        flush=True,
    )

    rng = np.random.default_rng(seed + 12345)
    global_step = 0
    update_index = 0
    best_score = -float("inf")
    latest_update: dict[str, float] = {}
    current_context: dict[str, Any] = {
        "algorithm": algorithm,
        "scenario": scenario,
        "episode": 0,
        "step": 0,
    }
    train_started = time.perf_counter()

    try:
        for episode in range(1, total_episodes + 1):
            current_context["episode"] = episode
            episode_started = time.perf_counter()
            episode_update_start = update_index
            obs_dict, _ = env.reset(seed=seed + episode - 1)
            obs = _obs_array(obs_dict, env.agents)
            ep_returns = np.zeros(env.n_agents, dtype=np.float64)
            comm_rates: list[float] = []
            saturations: list[float] = []
            reward_component_sums = {
                "communication": 0.0, "energy": 0.0, "safety": 0.0, "task": 0.0
            }
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
                # Truncation stops collection below but must not suppress Q bootstrapping.
                dones = _replay_terminal_mask(terminated, env.agents)
                algo.store(obs, action, rewards, next_obs, dones)
                ep_returns += rewards
                obs = next_obs
                global_step += 1
                comm_rates.append(float(last_info["mean_comm_rate_mbps"]))
                saturations.append(float(last_info["action_saturation"]))
                step_components = last_info.get("reward_components_sum", {})
                for component in ("communication", "energy", "safety", "task"):
                    reward_component_sums[component] += float(step_components.get(component, 0.0))

                if (
                    global_step >= int(cfg["runtime"]["update_after"])
                    and global_step % int(cfg["runtime"]["update_every"]) == 0
                ):
                    for _ in range(int(cfg["runtime"]["gradient_steps"])):
                        metrics = algo.update()
                        if metrics:
                            update_index += 1
                            latest_update = metrics
                            update_row = {"update": update_index, "global_step": global_step, **metrics}
                            logger.log_update(update_row)
                            wandb_update_every = int(cfg["runtime"].get("wandb_update_every", 100))
                            if update_index % max(1, wandb_update_every) == 0:
                                wb.log({f"update/{k}": v for k, v in update_row.items()})
                if all(truncated.values()) or all(terminated.values()):
                    break

            episode_sec = max(time.perf_counter() - episode_started, 1e-12)
            wall_time_sec = time.perf_counter() - train_started
            episode_updates = update_index - episode_update_start
            episode_steps = int(current_context.get("step", 0))
            ep_metrics: dict[str, Any] = {
                "episode": episode,
                "global_step": global_step,
                "episode_sec": float(episode_sec),
                "wall_time_sec": float(wall_time_sec),
                "env_steps_per_sec": float(episode_steps / episode_sec),
                "updates_per_sec": float(episode_updates / episode_sec),
                "updates": int(episode_updates),
                "return_mean": float(ep_returns.mean()),
                "return_sum": float(ep_returns.sum()),
                "search_rate": float(last_info.get("search_rate", 0.0)),
                "targets_found": int(last_info.get("targets_found", 0)),
                "energy_j": float(env.total_energy_used_j),
                "energy_consumption_pct": float(last_info.get("energy_consumption_pct", 0.0)),
                "mean_broken_link_s": float(last_info.get("mean_broken_link_s", 0.0)),
                "max_broken_link_s": float(last_info.get("max_broken_link_s", 0.0)),
                "safety_distance_violation_uavs": int(last_info.get("safety_distance_violation_uavs", 0)),
                "obstacle_hit_uavs": int(last_info.get("obstacle_hit_uavs", 0)),
                "boundary_hit_uavs": int(last_info.get("boundary_hit_uavs", 0)),
                "broken_link_uavs": int(last_info.get("broken_link_uavs", 0)),
                "depleted_uavs": int(last_info.get("depleted_uavs", 0)),
                "avg_battery_pct": float(last_info.get("avg_battery_pct", 100.0)),
                "action_saturation": float(np.mean(saturations)) if saturations else 0.0,
                "mean_comm_rate_mbps": float(np.mean(comm_rates)) if comm_rates else 0.0,
                "mission_delivery_rate": float(last_info.get("mission_delivery_rate", 0.0)),
                "reports_delivered": int(last_info.get("reports_delivered", 0)),
                "expired_reports": int(last_info.get("expired_reports", 0)),
                "queue_bytes_total": int(last_info.get("queue_bytes_total", 0)),
                "network_byte_pdr": float(last_info.get("network_byte_pdr", 0.0)),
                "network_throughput_bps": float(last_info.get("network_throughput_bps", 0.0)),
                "direct_gcs_uavs": int(last_info.get("direct_gcs_uavs", 0)),
                "multihop_gcs_uavs": int(last_info.get("multihop_gcs_uavs", 0)),
                "disconnected_gcs_uavs": int(last_info.get("disconnected_gcs_uavs", 0)),
                "mean_gcs_hops": float(last_info.get("mean_gcs_hops", 0.0)),
                "battery_capacity_j": float(last_info.get("battery_capacity_j", 0.0)),
                "critic_loss": float(latest_update.get("critic_loss", 0.0)),
                "critic1_loss": float(latest_update.get("critic1_loss", 0.0)),
                "critic2_loss": float(latest_update.get("critic2_loss", 0.0)),
                "actor_loss": float(latest_update.get("actor_loss", 0.0)),
                "q_mean": float(latest_update.get("q_mean", 0.0)),
                "target_q_mean": float(latest_update.get("target_q_mean", 0.0)),
                "td_error_abs_mean": float(latest_update.get("td_error_abs_mean", 0.0)),
                "q_gap_abs_mean": float(latest_update.get("q_gap_abs_mean", 0.0)),
                "entropy": float(latest_update.get("entropy", 0.0)),
                "alpha": float(latest_update.get("alpha", 0.0)),
                "alpha_loss": float(latest_update.get("alpha_loss", 0.0)),
            }
            ep_metrics.update(_sensing_metrics_from_info(last_info))
            fixed_returns = ep_returns[env.fixed_indices] if env.fixed_indices else np.asarray([], dtype=float)
            rotor_returns = ep_returns[env.multirotor_indices] if env.multirotor_indices else np.asarray([], dtype=float)
            ep_metrics["fixed_return_mean"] = float(fixed_returns.mean()) if len(fixed_returns) else 0.0
            ep_metrics["rotor_return_mean"] = float(rotor_returns.mean()) if len(rotor_returns) else 0.0
            ep_metrics["rotor_return_min"] = float(rotor_returns.min()) if len(rotor_returns) else 0.0
            for component in ("task", "communication", "energy", "safety"):
                ep_metrics[f"reward_{component}_sum"] = float(reward_component_sums[component])
            warmup_steps = int(cfg["runtime"]["warmup_steps"])
            if global_step <= warmup_steps:
                ep_metrics["phase"] = "warmup"
            elif episode < 30000:
                ep_metrics["phase"] = "learning"
            else:
                ep_metrics["phase"] = "post_convergence_reference"
            ep_metrics.update(diagnose_episode(ep_metrics))
            if profile.device.type == "cuda":
                ep_metrics["gpu_allocated_mb"] = float(
                    torch.cuda.memory_allocated(profile.device) / (1024**2)
                )
                ep_metrics["gpu_reserved_mb"] = float(
                    torch.cuda.memory_reserved(profile.device) / (1024**2)
                )
                ep_metrics["gpu_peak_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(profile.device) / (1024**2)
                )
            performance_metrics = {
                key: ep_metrics[key]
                for key in (
                    "episode",
                    "global_step",
                    "episode_sec",
                    "wall_time_sec",
                    "env_steps_per_sec",
                    "updates_per_sec",
                    "updates",
                    "gpu_allocated_mb",
                    "gpu_reserved_mb",
                    "gpu_peak_allocated_mb",
                )
                if key in ep_metrics
            }
            logger.log_performance(performance_metrics)
            low_payload = logger.log_episode(ep_metrics)
            if low_payload is not None:
                wb.log_low_episode(low_payload)
            # Keep W&B compact: one aggregate payload per episode. Detailed raw
            # optimizer updates remain in local CSV and are mirrored only every N updates.
            wandb_episode = {
                "paper/episode": episode,
                "paper/reward_total": ep_metrics["return_sum"],
                "paper/targets_found": ep_metrics["targets_found"],
                "paper/search_rate": ep_metrics["search_rate"],
                "paper/energy_consumption_pct": ep_metrics["energy_consumption_pct"],
                "paper/broken_link_duration_s": ep_metrics["mean_broken_link_s"],
                "swarm/episode": episode,
                "swarm/avg_battery_pct": ep_metrics["avg_battery_pct"],
                "swarm/battery_capacity_j": ep_metrics["battery_capacity_j"],
                "swarm/mean_altitude_m": ep_metrics["mean_altitude_m"],
                "swarm/min_altitude_m": ep_metrics["min_altitude_m"],
                "swarm/max_altitude_m": ep_metrics["max_altitude_m"],
                "swarm/depleted_uavs": ep_metrics["depleted_uavs"],
                "swarm/safety_distance_violation_uavs": ep_metrics["safety_distance_violation_uavs"],
                "swarm/obstacle_hit_uavs": ep_metrics["obstacle_hit_uavs"],
                "swarm/boundary_hit_uavs": ep_metrics["boundary_hit_uavs"],
                "swarm/broken_link_uavs": ep_metrics["broken_link_uavs"],
                "swarm/avg_comm_rate_mbps": ep_metrics["mean_comm_rate_mbps"],
                "swarm/avg_broken_link_s": ep_metrics["mean_broken_link_s"],
                "mission/delivery_rate": ep_metrics["mission_delivery_rate"],
                "mission/reports_delivered": ep_metrics["reports_delivered"],
                "mission/expired_reports": ep_metrics["expired_reports"],
                "network/byte_pdr": ep_metrics["network_byte_pdr"],
                "network/throughput_bps": ep_metrics["network_throughput_bps"],
                "network/direct_gcs_uavs": ep_metrics["direct_gcs_uavs"],
                "network/multihop_gcs_uavs": ep_metrics["multihop_gcs_uavs"],
                "network/disconnected_gcs_uavs": ep_metrics["disconnected_gcs_uavs"],
                "network/mean_gcs_hops": ep_metrics["mean_gcs_hops"],
                "network/queue_bytes_total": ep_metrics["queue_bytes_total"],
                "sensing/mean_belief_entropy": ep_metrics["mean_belief_entropy"],
                "sensing/mean_target_posterior": ep_metrics["mean_target_posterior"],
                "sensing/scanned_cells_total": ep_metrics["scanned_cells_total"],
                "sensing/positive_observations_total": ep_metrics["positive_sensor_observations_total"],
                "sensing/information_gain_total": ep_metrics["information_gain_total"],
                "sensing/targets_confirmed_total": ep_metrics["targets_confirmed_total"],
                "sensing/false_confirmations_total": ep_metrics["false_confirmations_total"],
                "sensing/confirmed_cells_total": ep_metrics["confirmed_cells_total"],
                "group/episode": episode,
                "group/fixed_return_mean": ep_metrics["fixed_return_mean"],
                "group/rotor_return_mean": ep_metrics["rotor_return_mean"],
                "reward/episode": episode,
                "reward/task": ep_metrics["reward_task_sum"],
                "reward/communication": ep_metrics["reward_communication_sum"],
                "reward/energy": ep_metrics["reward_energy_sum"],
                "reward/safety": ep_metrics["reward_safety_sum"],
                "rl/episode": episode,
                "rl/actor_loss": ep_metrics["actor_loss"],
                "rl/critic_loss": ep_metrics["critic_loss"],
                "rl/q_mean": ep_metrics["q_mean"],
                "rl/td_error": ep_metrics["td_error_abs_mean"],
                "performance/episode": episode,
                "performance/env_steps_per_sec": ep_metrics["env_steps_per_sec"],
                "performance/updates_per_sec": ep_metrics["updates_per_sec"],
                "performance/episode_sec": ep_metrics["episode_sec"],
            }
            if algorithm.lower() == "masac":
                wandb_episode["rl/entropy"] = ep_metrics["entropy"]
                wandb_episode["rl/alpha"] = ep_metrics["alpha"]
            wb.log(wandb_episode)

            if should_report_episode(episode, total_episodes, int(progress_every)):
                print(format_episode_progress(ep_metrics, total_episodes), flush=True)

            mission_score = (
                ep_metrics["mission_delivery_rate"] if env.peer_mode else ep_metrics["search_rate"]
            )
            score = ep_metrics["return_mean"] + 100.0 * mission_score
            if cfg["runtime"].get("save_best", True) and score > best_score:
                best_score = score
                algo.save(logger.checkpoint_dir / "best.pt")
            checkpoint_every = int(cfg["runtime"].get("checkpoint_every_episodes", 0))
            if checkpoint_every > 0 and episode % checkpoint_every == 0:
                latest_path = logger.checkpoint_dir / "latest.pt"
                algo.save(latest_path)
                wb.log_model(
                    latest_path,
                    f"{algorithm}-{scenario}-periodic",
                    aliases=["latest", f"episode-{episode}"],
                )
                # Scalar W&B panels are the primary live monitor. Refresh only the
                # three high-value visual summaries to keep media volume bounded.
                preview_paths = plot_training_curves(logger.episode_csv, logger.plots_dir)
                for preview in preview_paths:
                    if preview.name in {"reward.png", "group_returns.png", "reward_components.png"}:
                        wb.log_image(preview, f"live_plots/{preview.stem}")

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
            np.asarray(eval_env.trajectory),
            eval_env.targets,
            eval_env.obstacles,
            eval_env.agents,
            eval_env.agent_types,
            logger.plots_dir / "trajectory.png",
            eval_env.area_size_m,
        )
        plot_simulation_scenario(
            np.asarray(eval_env.trajectory)[0],
            eval_env.targets,
            eval_env.obstacles,
            eval_env.agents,
            eval_env.agent_types,
            eval_env.rotor_leaders,
            logger.plots_dir / "fig06_simulation_scenario.png",
            eval_env.area_size_m,
        )
        with (run_dir / "evaluation.json").open("w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
        total_training_sec = time.perf_counter() - train_started
        summary = {
            "algorithm": algorithm,
            "scenario": scenario,
            "episodes": total_episodes,
            "global_steps": global_step,
            "device": device,
            "best_score": best_score,
            "training_sec": float(total_training_sec),
            "tracking_mode": wb.mode,
            "wandb_project_url": wb.project_url,
            "wandb_run_url": wb.run_url,
            "runtime_profile": profile.as_dict(),
            "evaluation": eval_metrics,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        wb.update_summary(
            {
                "best_score": best_score,
                "training_sec": float(total_training_sec),
                "global_steps": global_step,
                "device": device,
            }
        )
        wb.log({f"eval/{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        for image_path in sorted(logger.plots_dir.glob("*.png")):
            wb.log_image(image_path, f"plots/{image_path.stem}")
        wb.log_model(
            logger.checkpoint_dir / "best.pt",
            f"{algorithm}-{scenario}-best",
            aliases=["best"],
        )
        wb.log_model(
            logger.checkpoint_dir / "final.pt",
            f"{algorithm}-{scenario}-final",
            aliases=["latest", "final"],
        )
        wb.log_run_artifact(f"{algorithm}-{scenario}-{name}-run")
        return run_dir
    except BaseException as exc:
        error_payload = logger.log_error(exc, current_context)
        wb.log_error(error_payload)
        try:
            crash_path = logger.checkpoint_dir / "crash.pt"
            algo.save(crash_path)
            wb.log_model(crash_path, f"{algorithm}-{scenario}-crash", aliases=["crash"])
        except BaseException as save_exc:
            save_payload = logger.log_error(
                save_exc,
                {**current_context, "during": "crash_checkpoint"},
            )
            wb.log_error(save_payload)
        wb.log_run_artifact(f"{algorithm}-{scenario}-{name}-crash-run")
        raise
    finally:
        wb.finish()
