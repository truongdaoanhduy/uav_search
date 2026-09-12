import numpy as np
import torch

from uav_search.algorithms.common import ReplayBuffer, soft_update
from uav_search.algorithms.networks import (
    CentralizedCritic,
    DeterministicActor,
    GaussianActor,
)


def test_replay_buffer_roundtrip_shapes():
    rb = ReplayBuffer(16, n_agents=3, obs_dim=7, action_dim=2, seed=1)
    for i in range(6):
        obs = np.full((3, 7), i, dtype=np.float32)
        act = np.zeros((3, 2), dtype=np.float32)
        rew = np.arange(3, dtype=np.float32)
        rb.add(obs, act, rew, obs + 1, np.zeros(3, dtype=np.float32))
    batch = rb.sample(4, torch.device("cpu"))
    assert batch.obs.shape == (4, 3, 7)
    assert batch.actions.shape == (4, 3, 2)
    assert batch.rewards.shape == (4, 3)
    assert batch.next_obs.shape == (4, 3, 7)
    assert batch.dones.shape == (4, 3)


def test_replay_buffer_uses_preallocated_torch_storage():
    rb = ReplayBuffer(16, n_agents=3, obs_dim=7, action_dim=2, seed=1)
    assert isinstance(rb.obs, torch.Tensor)
    assert isinstance(rb.actions, torch.Tensor)
    assert rb.obs.device.type == "cpu"
    assert rb.obs.dtype == torch.float32


def test_networks_return_bounded_actions_and_agent_qs():
    x = torch.randn(5, 7)
    det = DeterministicActor(7, 2, [32, 32])
    gauss = GaussianActor(7, 2, [32, 32], -5, 2)
    a_det = det(x)
    a_stoch, logp = gauss.sample(x)
    assert a_det.shape == (5, 2)
    assert a_stoch.shape == (5, 2)
    assert logp.shape == (5, 1)
    assert torch.all(a_det.abs() <= 1.0 + 1e-6)
    assert torch.all(a_stoch.abs() <= 1.0 + 1e-6)
    critic = CentralizedCritic(global_obs_dim=21, joint_action_dim=6, n_agents=3, hidden_sizes=[32, 32])
    q = critic(torch.randn(5, 21), torch.randn(5, 6))
    assert q.shape == (5, 3)


def test_soft_update_moves_target_toward_online():
    online = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        for p in online.parameters():
            p.fill_(1.0)
        for p in target.parameters():
            p.zero_()
    soft_update(target, online, 0.25)
    assert all(torch.allclose(p, torch.full_like(p, 0.25)) for p in target.parameters())