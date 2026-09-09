from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import logging
from typing import Any

import numpy as np

from .models import communication_rate_bps, path_gain_linear


GCS_NODE = -1
UAVNETSIM_REPOSITORY = "https://github.com/Zihao-Felix-Zhou/UavNetSim.git"
UAVNETSIM_COMMIT = "04daafb815eb377409b40b285574eeb62b9a8d58"
UAVNETSIM_VERSION = "2.0.0"


@dataclass(slots=True)
class NetworkLinkSnapshot:
    pair_rates_bps: np.ndarray
    gcs_rates_bps: np.ndarray
    adjacency: np.ndarray


@dataclass(slots=True, frozen=True)
class TransmissionIntent:
    sender: int
    recipient: int
    requested_bytes: int
    tx_power_w: float | None = None


@dataclass(slots=True, frozen=True)
class TransmissionOutcome:
    sender: int
    recipient: int
    requested_bytes: int
    delivered_bytes: int
    rate_bps: float
    distance_m: float
    delay_s: float = 0.0
    tx_energy_j: float = 0.0

    @property
    def success(self) -> bool:
        return self.delivered_bytes > 0


@dataclass(slots=True)
class NetworkStepResult:
    outcomes: list[TransmissionOutcome] = field(default_factory=list)
    attempted_bytes: int = 0
    delivered_bytes: int = 0
    byte_pdr: float = 0.0
    throughput_bps: float = 0.0
    mean_delay_s: float = 0.0
    phy_failures: int = 0
    # UAV-side radio TX energy for this macro-step. GCS energy is intentionally
    # excluded because only UAV batteries are part of the mission state.
    tx_energy_j: float = 0.0
    node_tx_energy_j: dict[int, float] = field(default_factory=dict)


class AnalyticalNetworkBackend:
    name = "analytical"

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.paper = cfg["paper"]
        self.assumed = cfg["assumed"]
        self.scenario = cfg.get("scenario", {})

    def _contact_range_m(self, *, gcs: bool = False) -> float:
        key = "gcs_contact_range_m" if gcs else "peer_contact_range_m"
        return float(self.scenario.get(key, float("inf")))

    def _tx_power_w(self) -> float:
        return float(
            self.scenario.get(
                "tx_power_reference_w",
                self.cfg.get("reference_backed", {}).get(
                    "tx_power_w", self.assumed.get("tx_power_w", 5.0)
                ),
            )
        )

    def _effective_contact_range_m(self, tx_power_w: float, *, gcs: bool = False) -> float:
        """Power-scaled operational reach around the calibrated reference range.

        Under free-space loss, received power scales as P_tx / d^2, so a fixed
        receiver threshold gives d_max proportional to sqrt(P_tx).  The base
        range remains an explicit u6 calibration parameter.
        """
        base = self._contact_range_m(gcs=gcs)
        reference = max(float(self.scenario.get("tx_power_reference_w", self._tx_power_w())), 1e-12)
        return float(base * np.sqrt(max(float(tx_power_w), 0.0) / reference))

    def _received_power_w(self, receiver_pos: np.ndarray, transmitter_pos: np.ndarray) -> float:
        delta = transmitter_pos - receiver_pos
        gain = path_gain_linear(
            float(np.linalg.norm(delta[:2])),
            float(delta[2]),
            self.cfg,
            force_los=False,
        )
        return float(self._tx_power_w() * gain)

    def link_snapshot(
        self,
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None = None,
    ) -> NetworkLinkSnapshot:
        del obstacles  # The retained analytical paper channel did not use building geometry.
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        n_agents = int(positions.shape[0])
        pair_rates = np.zeros((n_agents, n_agents), dtype=np.float64)
        adjacency = np.zeros((n_agents, n_agents), dtype=np.int8)
        gcs_rates = np.zeros(n_agents, dtype=np.float64)
        rmin = float(self.paper["min_comm_rate_bps"])

        for receiver in range(n_agents):
            for transmitter in range(n_agents):
                if receiver == transmitter:
                    continue
                distance = float(np.linalg.norm(positions[transmitter] - positions[receiver]))
                if distance > self._contact_range_m(gcs=False):
                    continue
                interference = 0.0
                for interferer in range(n_agents):
                    if interferer in (receiver, transmitter):
                        continue
                    interference += self._received_power_w(positions[receiver], positions[interferer])
                delta = positions[transmitter] - positions[receiver]
                rate = communication_rate_bps(
                    float(np.linalg.norm(delta[:2])),
                    float(delta[2]),
                    self.cfg,
                    interference_power_w=float(interference),
                    force_los=False,
                )
                pair_rates[receiver, transmitter] = rate
                if rate > rmin:
                    adjacency[receiver, transmitter] = 1

        for transmitter in range(n_agents):
            if float(np.linalg.norm(positions[transmitter] - gcs_position)) > self._contact_range_m(gcs=True):
                gcs_rates[transmitter] = 0.0
                continue
            interference = sum(
                self._received_power_w(gcs_position, positions[j])
                for j in range(n_agents)
                if j != transmitter
            )
            delta = positions[transmitter] - gcs_position
            gcs_rates[transmitter] = communication_rate_bps(
                float(np.linalg.norm(delta[:2])),
                float(delta[2]),
                self.cfg,
                interference_power_w=float(interference),
                force_los=False,
            )

        return NetworkLinkSnapshot(
            pair_rates_bps=pair_rates,
            gcs_rates_bps=gcs_rates,
            adjacency=adjacency,
        )



    def transmit(
        self,
        intents: list[TransmissionIntent],
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None,
        *,
        dt_s: float,
        step_index: int,
    ) -> NetworkStepResult:
        del step_index
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        snapshot = self.link_snapshot(positions, gcs_position, obstacles)
        outcomes: list[TransmissionOutcome] = []
        attempted = 0
        delivered = 0
        delays: list[float] = []
        tx_energy = 0.0
        node_tx_energy: dict[int, float] = {}
        for intent in intents:
            requested = max(0, int(intent.requested_bytes))
            if requested <= 0:
                continue
            attempted += requested
            sender = int(intent.sender)
            recipient = int(intent.recipient)
            tx_power_w = self._tx_power_w() if intent.tx_power_w is None else max(float(intent.tx_power_w), 0.0)
            if recipient == GCS_NODE:
                distance = float(np.linalg.norm(positions[sender] - gcs_position))
                delta = positions[sender] - gcs_position
                interference = sum(
                    self._received_power_w(gcs_position, positions[j])
                    for j in range(len(positions)) if j != sender
                )
                rate = communication_rate_bps(
                    float(np.linalg.norm(delta[:2])), float(delta[2]), self.cfg,
                    interference_power_w=float(interference), force_los=False,
                    tx_power_w=tx_power_w,
                )
                if distance > self._effective_contact_range_m(tx_power_w, gcs=True):
                    rate = 0.0
            else:
                distance = float(np.linalg.norm(positions[sender] - positions[recipient]))
                delta = positions[sender] - positions[recipient]
                interference = sum(
                    self._received_power_w(positions[recipient], positions[j])
                    for j in range(len(positions)) if j not in (sender, recipient)
                )
                rate = communication_rate_bps(
                    float(np.linalg.norm(delta[:2])), float(delta[2]), self.cfg,
                    interference_power_w=float(interference), force_los=False,
                    tx_power_w=tx_power_w,
                )
                if distance > self._effective_contact_range_m(tx_power_w, gcs=False):
                    rate = 0.0
            capacity = int(max(0.0, rate) * max(float(dt_s), 0.0) / 8.0)
            sent = min(requested, capacity) if rate > float(self.paper["min_comm_rate_bps"]) else 0
            delay = float(sent * 8.0 / rate) if sent > 0 and rate > 0 else 0.0
            energy = tx_power_w * delay
            delivered += sent
            if sent > 0:
                delays.append(delay)
            tx_energy += energy
            node_tx_energy[sender] = node_tx_energy.get(sender, 0.0) + float(energy)
            outcomes.append(TransmissionOutcome(
                sender=sender,
                recipient=recipient,
                requested_bytes=requested,
                delivered_bytes=sent,
                rate_bps=rate,
                distance_m=distance,
                delay_s=delay,
                tx_energy_j=energy,
            ))
        return NetworkStepResult(
            outcomes=outcomes,
            attempted_bytes=attempted,
            delivered_bytes=delivered,
            byte_pdr=float(delivered / attempted) if attempted else 0.0,
            throughput_bps=float(delivered * 8.0 / dt_s) if dt_s > 0 else 0.0,
            mean_delay_s=float(np.mean(delays)) if delays else 0.0,
            phy_failures=sum(1 for outcome in outcomes if outcome.requested_bytes > 0 and not outcome.success),
            tx_energy_j=float(tx_energy),
            node_tx_energy_j=node_tx_energy,
        )


class _CaptureEventBus:
    def __init__(self):
        self.events: list[tuple[str, float, dict[str, Any]]] = []

    def publish(self, event_type: str, sim_time_us: float, **data: Any) -> None:
        self.events.append((str(event_type), float(sim_time_us), dict(data)))


class _NetworkMetrics:
    def __init__(self):
        self.phy_successes = 0
        self.phy_failures = 0
        self.interference_events = 0
        # Native UavNetSim CSMA/CA records terminal ACK timeout service delay here.
        self.mac_delay: list[float] = []

    def record_phy_result(self, success: bool, interfered: bool) -> None:
        if success:
            self.phy_successes += 1
        else:
            self.phy_failures += 1
        if interfered:
            self.interference_events += 1


class _PaperAirspace:
    """Minimal UavNetSim airspace adapter backed by paper circular obstacles."""

    def __init__(self, obstacles: np.ndarray | None):
        raw = np.empty((0, 3), dtype=np.float64) if obstacles is None else np.asarray(obstacles, dtype=np.float64)
        self.obstacles = raw.reshape((-1, 3)) if raw.size else np.empty((0, 3), dtype=np.float64)

    @staticmethod
    def _segment_circle_intersects(a: np.ndarray, b: np.ndarray, circle: np.ndarray) -> bool:
        delta = b - a
        denom = float(np.dot(delta, delta))
        if denom <= 1e-12:
            return bool(np.linalg.norm(a - circle[:2]) <= circle[2])
        t = float(np.clip(np.dot(circle[:2] - a, delta) / denom, 0.0, 1.0))
        closest = a + t * delta
        return bool(np.linalg.norm(closest - circle[:2]) <= circle[2])

    def has_line_of_sight(self, start: Any, end: Any) -> bool:
        a = np.asarray(start, dtype=np.float64)[:2]
        b = np.asarray(end, dtype=np.float64)[:2]
        return not any(self._segment_circle_intersects(a, b, circle) for circle in self.obstacles)


class _PolicyHopRoutingAdapter:
    """Minimal network-layer adapter for native UavNetSim ACK/ARQ.

    MARL remains the sole next-hop authority: this adapter never selects or
    forwards a data packet to another hop.  It only acknowledges a data packet
    that the UavNetSim PHY delivered to the policy-selected receiver and handles
    ACK reception at the sender so CSMA/CA can perform its native retry logic.
    """

    def __init__(self, node: "_RadioNode", packet_module: Any, config_module: Any):
        self.my_drone = node
        self.simulator = node.simulator
        self.env = node.env
        self.packet_module = packet_module
        self.config = config_module

    def penalize(self, packet: Any) -> None:
        self.simulator.event_bus.publish(
            "packet_ack_timeout",
            self.env.now,
            packet_id=int(packet.packet_id),
            sender=int(self.my_drone.identifier),
        )

    def packet_reception(self, packet: Any, src_drone_id: int):
        DataPacket = self.packet_module.DataPacket
        AckPacket = self.packet_module.AckPacket
        if isinstance(packet, DataPacket):
            self.simulator.ack_sequence += 1
            ack_packet = AckPacket(
                src_drone=self.my_drone,
                dst_drone=self.simulator.drones[int(src_drone_id)],
                ack_packet_id=int(self.simulator.ack_sequence),
                ack_packet_length=int(self.config.ACK_PACKET_LENGTH),
                ack_packet=packet,
                simulator=self.simulator,
                channel_id=int(packet.channel_id),
                creation_time=self.env.now,
            )
            # Native UavNetSim stop-and-wait behavior: ACK is transmitted after
            # SIFS and bypasses CSMA contention.  No application-level forwarding
            # occurs here; forwarding requires a future MARL action.
            yield self.env.timeout(float(self.config.SIFS_DURATION))
            if not self.my_drone.sleep:
                ack_packet.increase_ttl()
                self.my_drone.mac_protocol.phy.unicast(ack_packet, int(src_drone_id))
                yield self.env.timeout(
                    float(ack_packet.packet_length) / float(self.config.BIT_RATE) * 1e6
                )
            return

        if isinstance(packet, AckPacket):
            data_packet = packet.ack_packet
            key = f"wait_ack{self.my_drone.identifier}_{data_packet.packet_id}"
            if self.my_drone.mac_protocol.wait_ack_process_finish.get(key, 1) != 0:
                return
            wait_process = self.my_drone.mac_protocol.wait_ack_process_dict.get(key)
            if wait_process is None or wait_process.triggered:
                return
            self.my_drone.mac_protocol.wait_ack_process_finish[key] = 1
            self.simulator.event_bus.publish(
                "packet_ack_received",
                self.env.now,
                packet_id=int(data_packet.packet_id),
                sender=int(self.my_drone.identifier),
                ack_from=int(src_drone_id),
            )
            wait_process.interrupt()


class _RadioNode:
    def __init__(self, simulator: Any, identifier: int, coords: np.ndarray):
        self.simulator = simulator
        self.env = simulator.env
        self.identifier = int(identifier)
        self.coords = [float(value) for value in coords]
        self.residual_energy = 1.0e12
        self.mac_process_dict: dict[str, Any] = {}
        self.mac_process_finish: dict[str, int] = {}
        self.sleep = False
        self.routing_protocol: Any | None = None

    def packet_coming(self, packet: Any):
        """Re-enter native CSMA/CA after an ACK timeout."""
        if self.sleep:
            return
        packet.number_retransmission_attempt[self.identifier] += 1
        key = f"mac_send{self.identifier}_{packet.packet_id}"
        self.mac_process_finish[key] = 0
        process = self.env.process(self.mac_protocol.mac_send(packet))
        self.mac_process_dict[key] = process
        yield process

    def receive(self):
        """Deliver successful PHY receptions to the ACK-only routing adapter."""
        while True:
            reception = yield self.inbox.get()
            if self.sleep:
                continue
            packet = reception.packet
            if packet.get_current_ttl() >= self.simulator.n_drones + 1:
                self.simulator.event_bus.publish(
                    "packet_dropped",
                    self.env.now,
                    packet_id=int(packet.packet_id),
                    node=int(self.identifier),
                    reason="ttl_exceeded",
                )
                continue
            yield self.env.process(
                self.routing_protocol.packet_reception(packet, int(reception.transmitter_id))
            )


class UavNetSimBackend:
    """Peer transport using UavNetSim's A2A PHY, channel and CSMA/CA implementation.

    PaperUAVEnv remains authoritative for mobility, application queues and MARL
    routing decisions.  This adapter intentionally never calls a UavNetSim
    routing protocol, so a policy-selected recipient cannot be overwritten.
    """

    name = "uavnetsim"

    def __init__(self, cfg: dict[str, Any], seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.paper = cfg["paper"]
        self.assumed = cfg["assumed"]
        self.scenario = cfg.get("scenario", {})
        self._modules = self._load_modules()
        self._episode: Any | None = None
        self._episode_counter = 0
        self._packet_sequence = 0

    @staticmethod
    def _load_modules() -> dict[str, Any]:
        # Upstream simulator/log.py configures a root FileHandler at import time if
        # the process has no logging handlers. Install a temporary NullHandler so
        # importing UavNetSim cannot create running_log.log inside the experiment repo.
        root_logger = logging.getLogger()
        import_guard = logging.NullHandler()
        added_guard = not root_logger.handlers
        if added_guard:
            root_logger.addHandler(import_guard)
        try:
            return {
                "simpy": importlib.import_module("simpy"),
                "config": importlib.import_module("utils.config"),
                "channel": importlib.import_module("phy.channel"),
                "csma": importlib.import_module("mac.csma_ca"),
                "packet": importlib.import_module("entities.packet"),
                "a2a": importlib.import_module("phy.a2a"),
            }
        except (ModuleNotFoundError, ImportError) as exc:
            raise RuntimeError(
                "UavNetSim backend is optional and is not installed. "
                "Run scripts/install_uavnetsim.sh to install the pinned lightweight "
                f"UavNetSim {UAVNETSIM_VERSION} commit {UAVNETSIM_COMMIT}."
            ) from exc
        finally:
            if added_guard:
                root_logger.removeHandler(import_guard)

    def _setting(self, key: str, default: Any) -> Any:
        return self.scenario.get(key, default)

    def _contact_range_m(self, *, gcs: bool = False) -> float:
        key = "gcs_contact_range_m" if gcs else "peer_contact_range_m"
        return float(self._setting(key, float("inf")))

    def _radio_parameters(self) -> dict[str, float | str]:
        ref = self.cfg.get("reference_backed", {})
        ucfg = self._modules["config"]
        return {
            "CHANNEL_MODE": "a2a",
            "LOS_A2A_MODEL": str(self._setting("uavnetsim_los_model", "free_space")),
            "NLOS_A2A_MODEL": str(self._setting("uavnetsim_nlos_model", "urban")),
            "CARRIER_FREQUENCY": float(self._setting("uavnetsim_carrier_hz", ref.get("carrier_hz", ucfg.CARRIER_FREQUENCY))),
            "BANDWIDTH": float(self._setting("uavnetsim_bandwidth_hz", ref.get("bandwidth_hz", ucfg.BANDWIDTH))),
            "BIT_RATE": float(self._setting("uavnetsim_bit_rate_bps", 2_000_000.0)),
            "TRANSMITTING_POWER": float(self._setting("uavnetsim_tx_power_w", self.paper.get("communication_power_w", 5.0))),
            "SINR_THRESHOLD_DB": float(self._setting("uavnetsim_sinr_threshold_db", 6.0)),
        }

    def _with_radio_parameters(self):
        backend = self

        class _ConfigScope:
            def __enter__(self_nonlocal):
                ucfg = backend._modules["config"]
                self_nonlocal.previous = {}
                for key, value in backend._radio_parameters().items():
                    self_nonlocal.previous[key] = getattr(ucfg, key)
                    setattr(ucfg, key, value)
                ucfg.BIT_TRANSMISSION_TIME = 1.0 / float(ucfg.BIT_RATE) * 1e6
                return ucfg

            def __exit__(self_nonlocal, exc_type, exc, tb):
                ucfg = backend._modules["config"]
                for key, value in self_nonlocal.previous.items():
                    setattr(ucfg, key, value)
                ucfg.BIT_TRANSMISSION_TIME = 1.0 / float(ucfg.BIT_RATE) * 1e6
                return False

        return _ConfigScope()

    @property
    def bit_rate_bps(self) -> float:
        return float(self._radio_parameters()["BIT_RATE"])

    @property
    def episode_instance_id(self) -> int:
        return int(self._episode_counter)

    @property
    def simulation_time_s(self) -> float:
        if self._episode is None:
            return 0.0
        return float(self._episode.env.now) / 1e6

    def set_node_active(self, active_mask: np.ndarray) -> None:
        """Synchronize mission battery-death state into persistent radio nodes."""
        if self._episode is None:
            return
        active = np.asarray(active_mask, dtype=bool).reshape(-1)
        n_uavs = max(0, len(self._episode.drones) - 1)
        if active.size != n_uavs:
            raise ValueError(f"active_mask has {active.size} entries; expected {n_uavs}")
        for node_id in range(n_uavs):
            self._episode.drones[node_id].sleep = not bool(active[node_id])
        # The final synthetic node is the fixed GCS sink and is not battery-limited.
        self._episode.drones[n_uavs].sleep = False

    def reset_episode(
        self,
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None = None,
    ) -> None:
        """Create one persistent SimPy/UavNetSim MAC/PHY instance for an episode."""
        if self._episode is not None:
            close = getattr(self._episode.channel, "close", None)
            if callable(close):
                close()
        simpy = self._modules["simpy"]
        BaseChannel = self._modules["channel"].Channel
        CsmaCa = self._modules["csma"].CsmaCa
        ucfg = self._modules["config"]
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        all_positions = np.vstack([positions, gcs_position])
        n_agents = int(len(positions))

        env = simpy.Environment()
        event_bus = _CaptureEventBus()
        metrics = _NetworkMetrics()
        simulator = type("UavNetSimEpisodeSimulator", (), {})()
        simulator.env = env
        simulator.seed = int(self.seed)
        simulator.event_bus = event_bus
        simulator.metrics = metrics
        simulator.airspace = _PaperAirspace(obstacles)
        simulator.n_drones = n_agents + 1
        simulator.channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_agents + 1)}
        simulator.drones = [_RadioNode(simulator, i, all_positions[i]) for i in range(n_agents + 1)]
        simulator.ack_sequence = int((self._episode_counter + 1) * 1_000_000_000)

        class PowerAwareChannel(BaseChannel):
            def transmit(channel_self, packet, transmitter_id, receiver_ids):
                previous = float(ucfg.TRANSMITTING_POWER)
                try:
                    ucfg.TRANSMITTING_POWER = float(getattr(packet, "tx_power_w", previous))
                    return super(PowerAwareChannel, channel_self).transmit(packet, transmitter_id, receiver_ids)
                finally:
                    ucfg.TRANSMITTING_POWER = previous

        with self._with_radio_parameters():
            simulator.channel = PowerAwareChannel(env, simulator)
            for node in simulator.drones:
                node.inbox = simulator.channel.create_inbox_for_receiver(node.identifier)
                node.mac_protocol = CsmaCa(node)
                node.mac_protocol.enable_ack = True
                node.routing_protocol = _PolicyHopRoutingAdapter(
                    node, self._modules["packet"], ucfg
                )
                env.process(node.receive())
                phy = node.mac_protocol.phy

                def consume_energy(packet, *, _node=node, _ucfg=ucfg):
                    duration_s = float(packet.packet_length) / float(_ucfg.BIT_RATE)
                    power_w = max(0.0, float(getattr(packet, "tx_power_w", _ucfg.TRANSMITTING_POWER)))
                    energy_j = duration_s * power_w
                    _node.residual_energy = max(0.0, _node.residual_energy - energy_j)
                    _node.simulator.event_bus.publish(
                        "packet_tx_energy",
                        _node.env.now,
                        packet_id=int(packet.packet_id),
                        node=int(_node.identifier),
                        packet_type=packet.__class__.__name__,
                        energy_j=float(energy_j),
                    )

                phy._consume_transmit_energy = consume_energy

        self._episode = simulator
        self._episode_counter += 1
        self._packet_sequence = 0

    def _update_episode_geometry(
        self,
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None,
    ) -> Any:
        if self._episode is None:
            self.reset_episode(positions, gcs_position, obstacles)
        simulator = self._episode
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        all_positions = np.vstack([positions, gcs_position])
        if len(simulator.drones) != len(all_positions):
            self.reset_episode(positions, gcs_position, obstacles)
            simulator = self._episode
        for node, coords in zip(simulator.drones, all_positions):
            node.coords = [float(v) for v in coords]
        raw = np.empty((0, 3), dtype=np.float64) if obstacles is None else np.asarray(obstacles, dtype=np.float64)
        simulator.airspace.obstacles = raw.reshape((-1, 3)) if raw.size else np.empty((0, 3), dtype=np.float64)
        return simulator

    def _gain(self, start: np.ndarray, end: np.ndarray, airspace: _PaperAirspace) -> float:
        a2a = self._modules["a2a"]
        params = self._radio_parameters()
        los = airspace.has_line_of_sight(start, end)
        model = str(params["LOS_A2A_MODEL"] if los else params["NLOS_A2A_MODEL"])
        return float(a2a.path_gain(float(np.linalg.norm(end - start)), float(params["CARRIER_FREQUENCY"]), model, los))

    def _rate(self, start: np.ndarray, end: np.ndarray, airspace: _PaperAirspace) -> float:
        params = self._radio_parameters()
        gain = self._gain(start, end, airspace)
        signal = float(params["TRANSMITTING_POWER"]) * gain
        bandwidth = max(float(params["BANDWIDTH"]), 1.0)
        thermal_noise_density_dbm_hz = -174.0
        receiver_noise_figure_db = 7.0
        noise_dbm = thermal_noise_density_dbm_hz + 10.0 * np.log10(bandwidth) + receiver_noise_figure_db
        noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)
        sinr_db = 10.0 * np.log10(signal / noise_w) if signal > 0.0 else -200.0
        return float(params["BIT_RATE"]) if sinr_db >= float(params["SINR_THRESHOLD_DB"]) else 0.0

    def link_snapshot(
        self,
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None = None,
    ) -> NetworkLinkSnapshot:
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        airspace = _PaperAirspace(obstacles)
        n_agents = int(len(positions))
        pair_rates = np.zeros((n_agents, n_agents), dtype=np.float64)
        gcs_rates = np.zeros(n_agents, dtype=np.float64)
        adjacency = np.zeros((n_agents, n_agents), dtype=np.int8)
        rmin = float(self.paper["min_comm_rate_bps"])
        for receiver in range(n_agents):
            for transmitter in range(n_agents):
                if receiver == transmitter:
                    continue
                if float(np.linalg.norm(positions[transmitter] - positions[receiver])) > self._contact_range_m(gcs=False):
                    continue
                rate = self._rate(positions[transmitter], positions[receiver], airspace)
                pair_rates[receiver, transmitter] = rate
                if rate > rmin:
                    adjacency[receiver, transmitter] = 1
        for transmitter in range(n_agents):
            if float(np.linalg.norm(positions[transmitter] - gcs_position)) <= self._contact_range_m(gcs=True):
                gcs_rates[transmitter] = self._rate(positions[transmitter], gcs_position, airspace)
        return NetworkLinkSnapshot(pair_rates, gcs_rates, adjacency)

    def transmit(
        self,
        intents: list[TransmissionIntent],
        positions: np.ndarray,
        gcs_position: np.ndarray,
        obstacles: np.ndarray | None,
        *,
        dt_s: float,
        step_index: int,
    ) -> NetworkStepResult:
        intents = [intent for intent in intents if int(intent.requested_bytes) > 0]
        attempted = int(sum(int(intent.requested_bytes) for intent in intents))
        if not intents or dt_s <= 0.0:
            return NetworkStepResult(attempted_bytes=attempted)

        DataPacket = self._modules["packet"].DataPacket
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        n_agents = int(len(positions))
        gcs_id = n_agents
        simulator = self._update_episode_geometry(positions, gcs_position, obstacles)
        env = simulator.env
        event_bus = simulator.event_bus
        metrics = simulator.metrics
        valid_intents: list[TransmissionIntent] = []
        prefailed: list[TransmissionOutcome] = []
        airspace = simulator.airspace
        params = self._radio_parameters()
        noise_w = 10.0 ** (((-174.0 + 10.0 * np.log10(max(float(params["BANDWIDTH"]), 1.0)) + 7.0) - 30.0) / 10.0)
        for intent in intents:
            sender = int(intent.sender)
            recipient = int(intent.recipient)
            tx_power = max(0.0, float(params["TRANSMITTING_POWER"] if intent.tx_power_w is None else intent.tx_power_w))
            if sender < 0 or sender >= n_agents:
                continue
            if recipient == GCS_NODE:
                endpoint = gcs_position
                distance = float(np.linalg.norm(positions[sender] - endpoint))
                recipient_sleeping = False
            elif 0 <= recipient < n_agents and recipient != sender:
                endpoint = positions[recipient]
                distance = float(np.linalg.norm(positions[sender] - endpoint))
                recipient_sleeping = bool(simulator.drones[recipient].sleep)
            else:
                prefailed.append(TransmissionOutcome(sender, recipient, int(intent.requested_bytes), 0, 0.0, 0.0))
                continue
            if bool(simulator.drones[sender].sleep) or recipient_sleeping:
                prefailed.append(TransmissionOutcome(
                    sender, recipient, int(intent.requested_bytes), 0, 0.0, distance
                ))
                continue
            gain = self._gain(positions[sender], endpoint, airspace)
            sinr_db = 10.0 * np.log10(tx_power * gain / noise_w) if tx_power > 0.0 else -200.0
            rate = float(params["BIT_RATE"]) if sinr_db >= float(params["SINR_THRESHOLD_DB"]) else 0.0
            # Do not suppress a physically poor attempt before UavNetSim sees it:
            # a selected low-power/far transmission still spends airtime/energy.
            valid_intents.append(intent)

        if not valid_intents:
            return NetworkStepResult(
                outcomes=prefailed,
                attempted_bytes=attempted,
                delivered_bytes=0,
                byte_pdr=0.0,
                throughput_bps=0.0,
                mean_delay_s=0.0,
                phy_failures=len(prefailed),
                tx_energy_j=0.0,
            )

        packets: dict[int, tuple[Any, TransmissionIntent, int, int]] = {}
        phy_failures_before = int(metrics.phy_failures)
        event_start = len(event_bus.events)
        with self._with_radio_parameters() as ucfg:

            ip_header = int(getattr(ucfg, "IP_HEADER_LENGTH", 0))
            mac_header = int(getattr(ucfg, "MAC_HEADER_LENGTH", 0))
            phy_header = int(getattr(ucfg, "PHY_HEADER_LENGTH", 0))
            header_bits = ip_header + mac_header + phy_header
            slot_us = float(dt_s) * 1e6
            max_backoff_us = max(0, int(ucfg.CW_MIN) - 1) * float(ucfg.SLOT_DURATION)
            usable_tx_us = max(0.0, slot_us - float(ucfg.DIFS_DURATION) - max_backoff_us - 1.0)
            max_payload_bits = max(0, int(float(ucfg.BIT_RATE) * usable_tx_us / 1e6) - header_bits)

            packet_payload_bytes = max(
                1,
                int(self.scenario.get("network_packet_payload_bytes", max(1, int(getattr(ucfg, "AVERAGE_PAYLOAD_LENGTH", 8192)) // 8))),
            )
            intent_meta: dict[tuple[int, int], dict[str, float | int]] = {}
            for intent in valid_intents:
                sender = int(intent.sender)
                recipient_raw = int(intent.recipient)
                recipient = gcs_id if recipient_raw == GCS_NODE else recipient_raw
                if sender < 0 or sender >= n_agents or recipient < 0 or recipient > gcs_id or sender == recipient:
                    continue
                tx_power_w = max(0.0, float(ucfg.TRANSMITTING_POWER if intent.tx_power_w is None else intent.tx_power_w))
                endpoint = gcs_position if recipient_raw == GCS_NODE else positions[recipient_raw]
                distance = float(np.linalg.norm(positions[sender] - endpoint))
                gain = self._gain(positions[sender], endpoint, simulator.airspace)
                sinr_db = 10.0 * np.log10(tx_power_w * gain / noise_w) if tx_power_w > 0.0 else -200.0
                nominal_rate = float(ucfg.BIT_RATE) if sinr_db >= float(ucfg.SINR_THRESHOLD_DB) else 0.0
                key_meta = (sender, recipient_raw)
                intent_meta[key_meta] = {
                    "requested": int(intent.requested_bytes),
                    "distance": distance,
                    "rate": nominal_rate,
                    "tx_power_w": tx_power_w,
                }
                remaining = min(int(intent.requested_bytes), max(0, max_payload_bits // 8))
                while remaining > 0:
                    payload_bytes = min(remaining, packet_payload_bytes)
                    payload_bits = payload_bytes * 8
                    packet_length = max(1, header_bits + payload_bits)
                    self._packet_sequence += 1
                    packet_id = int(self._episode_counter * 1_000_000_000 + self._packet_sequence)
                    packet = DataPacket(
                        src_drone=simulator.drones[sender],
                        dst_drone=simulator.drones[recipient],
                        creation_time=env.now,
                        data_packet_id=packet_id,
                        data_packet_length=packet_length,
                        simulator=simulator,
                        channel_id=1,
                    )
                    packet.transmission_mode = 0
                    packet.next_hop_id = recipient
                    packet.tx_power_w = tx_power_w
                    packet.number_retransmission_attempt[sender] = 1
                    mac_key = f"mac_send{sender}_{packet_id}"
                    simulator.drones[sender].mac_process_finish[mac_key] = 0
                    proc = env.process(simulator.drones[sender].mac_protocol.mac_send(packet))
                    simulator.drones[sender].mac_process_dict[mac_key] = proc
                    packets[packet_id] = (packet, intent, recipient, payload_bytes)
                    remaining -= payload_bytes

            slot_end_us = float(env.now) + float(dt_s) * 1e6
            if env.peek() < float("inf"):
                env.run(until=slot_end_us)
            else:
                env.run(until=slot_end_us)

            # A hop is committed to the application layer only after native
            # UavNetSim ACK reception. Raw PHY delivery alone is insufficient:
            # if the ACK is lost, CSMA/CA retries and the report remains at the sender.
            success_events = {
                int(data["packet_id"]): (time_us, data)
                for event_type, time_us, data in event_bus.events
                if event_type == "packet_ack_received" and int(data.get("packet_id", -1)) in packets
            }
            delivered_by_intent: dict[tuple[int, int], int] = {key: 0 for key in intent_meta}
            delay_by_intent: dict[tuple[int, int], list[float]] = {key: [] for key in intent_meta}
            for packet_id, (packet, intent, _recipient, payload_bytes) in packets.items():
                success_event = success_events.get(packet_id)
                if success_event is None:
                    continue
                key_meta = (int(intent.sender), int(intent.recipient))
                delivered_by_intent[key_meta] += int(payload_bytes)
                delay_by_intent[key_meta].append(max(0.0, float(success_event[0] - packet.creation_time) / 1e6))

            outcomes: list[TransmissionOutcome] = list(prefailed)
            delays: list[float] = []
            delivered_total = 0
            step_events = event_bus.events[event_start:]
            node_tx_energy: dict[int, float] = {}
            data_energy_by_intent: dict[tuple[int, int], float] = {key: 0.0 for key in intent_meta}
            for event_type, _time_us, data in step_events:
                if event_type != "packet_tx_energy":
                    continue
                node_id = int(data.get("node", -1))
                energy_j = max(0.0, float(data.get("energy_j", 0.0)))
                if 0 <= node_id < n_agents:
                    node_tx_energy[node_id] = node_tx_energy.get(node_id, 0.0) + energy_j
                if data.get("packet_type") != "DataPacket":
                    continue
                packet_id = int(data.get("packet_id", -1))
                packet_meta = packets.get(packet_id)
                if packet_meta is None:
                    continue
                intent = packet_meta[1]
                key_meta = (int(intent.sender), int(intent.recipient))
                if key_meta in data_energy_by_intent:
                    data_energy_by_intent[key_meta] += energy_j
            for intent in valid_intents:
                sender = int(intent.sender)
                recipient_raw = int(intent.recipient)
                key_meta = (sender, recipient_raw)
                if key_meta not in intent_meta:
                    continue
                delivered = int(delivered_by_intent.get(key_meta, 0))
                delivered_total += delivered
                intent_delays = delay_by_intent.get(key_meta, [])
                delay_s = float(max(intent_delays)) if intent_delays else 0.0
                delays.extend(intent_delays)
                energy_j = float(data_energy_by_intent.get(key_meta, 0.0))
                meta = intent_meta[key_meta]
                outcomes.append(TransmissionOutcome(
                    sender=sender,
                    recipient=recipient_raw,
                    requested_bytes=int(intent.requested_bytes),
                    delivered_bytes=delivered,
                    rate_bps=float(meta["rate"]),
                    distance_m=float(meta["distance"]),
                    delay_s=delay_s,
                    tx_energy_j=energy_j,
                ))

        return NetworkStepResult(
            outcomes=outcomes,
            attempted_bytes=attempted,
            delivered_bytes=int(delivered_total),
            byte_pdr=float(delivered_total / attempted) if attempted else 0.0,
            throughput_bps=float(delivered_total * 8.0 / dt_s),
            mean_delay_s=float(np.mean(delays)) if delays else 0.0,
            phy_failures=max(0, int(metrics.phy_failures) - phy_failures_before) + len(prefailed),
            tx_energy_j=float(sum(node_tx_energy.values())),
            node_tx_energy_j=node_tx_energy,
        )

def create_network_backend(name: str, cfg: dict[str, Any], seed: int = 0):
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized == "analytical":
        return AnalyticalNetworkBackend(cfg, seed=seed)
    if normalized in {"uavnetsim", "uav_net_sim"}:
        return UavNetSimBackend(cfg, seed=seed)
    raise ValueError(f"Unsupported network backend: {name}")
