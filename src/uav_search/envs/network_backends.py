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
    tx_energy_j: float = 0.0


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
            self.cfg.get("reference_backed", {}).get(
                "tx_power_w", self.assumed.get("tx_power_w", 5.0)
            )
        )

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
        for intent in intents:
            requested = max(0, int(intent.requested_bytes))
            if requested <= 0:
                continue
            attempted += requested
            sender = int(intent.sender)
            recipient = int(intent.recipient)
            if recipient == GCS_NODE:
                rate = float(snapshot.gcs_rates_bps[sender])
                distance = float(np.linalg.norm(positions[sender] - gcs_position))
            else:
                rate = float(snapshot.pair_rates_bps[recipient, sender])
                distance = float(np.linalg.norm(positions[sender] - positions[recipient]))
            capacity = int(max(0.0, rate) * max(float(dt_s), 0.0) / 8.0)
            sent = min(requested, capacity) if rate > float(self.paper["min_comm_rate_bps"]) else 0
            delay = float(sent * 8.0 / rate) if sent > 0 and rate > 0 else 0.0
            energy = float(self.paper.get("communication_power_w", 0.0)) * delay
            delivered += sent
            if sent > 0:
                delays.append(delay)
            tx_energy += energy
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


class _RadioNode:
    def __init__(self, simulator: Any, identifier: int, coords: np.ndarray):
        self.simulator = simulator
        self.env = simulator.env
        self.identifier = int(identifier)
        self.coords = [float(value) for value in coords]
        self.residual_energy = 1.0e12
        self.mac_process_dict: dict[str, Any] = {}
        self.mac_process_finish: dict[str, int] = {}


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

        simpy = self._modules["simpy"]
        Channel = self._modules["channel"].Channel
        CsmaCa = self._modules["csma"].CsmaCa
        DataPacket = self._modules["packet"].DataPacket
        positions = np.asarray(positions, dtype=np.float64)
        gcs_position = np.asarray(gcs_position, dtype=np.float64)
        n_agents = int(len(positions))
        gcs_id = n_agents
        all_positions = np.vstack([positions, gcs_position])
        snapshot = self.link_snapshot(positions, gcs_position, obstacles)
        valid_intents: list[TransmissionIntent] = []
        prefailed: list[TransmissionOutcome] = []
        for intent in intents:
            sender = int(intent.sender)
            recipient = int(intent.recipient)
            if sender < 0 or sender >= n_agents:
                continue
            if recipient == GCS_NODE:
                rate = float(snapshot.gcs_rates_bps[sender])
                distance = float(np.linalg.norm(positions[sender] - gcs_position))
            elif 0 <= recipient < n_agents and recipient != sender:
                rate = float(snapshot.pair_rates_bps[recipient, sender])
                distance = float(np.linalg.norm(positions[sender] - positions[recipient]))
            else:
                rate = 0.0
                distance = 0.0
            if rate <= 0.0:
                prefailed.append(TransmissionOutcome(
                    sender=sender,
                    recipient=recipient,
                    requested_bytes=int(intent.requested_bytes),
                    delivered_bytes=0,
                    rate_bps=0.0,
                    distance_m=distance,
                    delay_s=0.0,
                    tx_energy_j=0.0,
                ))
            else:
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

        env = simpy.Environment()
        event_bus = _CaptureEventBus()
        metrics = _NetworkMetrics()
        simulator = type("UavNetSimMicroSimulator", (), {})()
        simulator.env = env
        simulator.seed = int(self.seed + 10_000 * int(step_index))
        simulator.event_bus = event_bus
        simulator.metrics = metrics
        simulator.airspace = _PaperAirspace(obstacles)
        simulator.n_drones = n_agents + 1
        simulator.channel_states = {i: simpy.Resource(env, capacity=1) for i in range(n_agents + 1)}
        simulator.drones = [_RadioNode(simulator, i, all_positions[i]) for i in range(n_agents + 1)]

        start_energy: dict[int, float] = {}
        packets: dict[int, tuple[Any, TransmissionIntent, int]] = {}
        with self._with_radio_parameters() as ucfg:
            simulator.channel = Channel(env, simulator)
            for node in simulator.drones:
                node.inbox = simulator.channel.create_inbox_for_receiver(node.identifier)
                node.mac_protocol = CsmaCa(node)
                node.mac_protocol.enable_ack = False

            ip_header = int(getattr(ucfg, "IP_HEADER_LENGTH", 0))
            mac_header = int(getattr(ucfg, "MAC_HEADER_LENGTH", 0))
            phy_header = int(getattr(ucfg, "PHY_HEADER_LENGTH", 0))
            header_bits = ip_header + mac_header + phy_header
            slot_us = float(dt_s) * 1e6
            max_backoff_us = max(0, int(ucfg.CW_MIN) - 1) * float(ucfg.SLOT_DURATION)
            usable_tx_us = max(0.0, slot_us - float(ucfg.DIFS_DURATION) - max_backoff_us - 1.0)
            max_payload_bits = max(0, int(float(ucfg.BIT_RATE) * usable_tx_us / 1e6) - header_bits)

            for sequence, intent in enumerate(valid_intents, start=1):
                sender = int(intent.sender)
                recipient = gcs_id if int(intent.recipient) == GCS_NODE else int(intent.recipient)
                if sender < 0 or sender >= n_agents or recipient < 0 or recipient > gcs_id or sender == recipient:
                    continue
                payload_bits = min(max_payload_bits, max(1, int(intent.requested_bytes) * 8))
                if payload_bits <= 0:
                    continue
                payload_bytes = min(int(intent.requested_bytes), payload_bits // 8)
                if payload_bytes <= 0:
                    continue
                payload_bits = payload_bytes * 8
                packet_length = max(1, header_bits + payload_bits)
                packet_id = int((step_index + 1) * 1_000_000 + sequence)
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
                packet.number_retransmission_attempt[sender] = 1
                key = f"mac_send{sender}_{packet_id}"
                simulator.drones[sender].mac_process_finish[key] = 0
                start_energy[sender] = simulator.drones[sender].residual_energy
                proc = env.process(simulator.drones[sender].mac_protocol.mac_send(packet))
                simulator.drones[sender].mac_process_dict[key] = proc
                packets[packet_id] = (packet, intent, recipient, payload_bytes)

            slot_end_us = float(dt_s) * 1e6
            if env.peek() < float("inf"):
                env.run(until=slot_end_us)

            success_events = {
                int(data["packet_id"]): (time_us, data)
                for event_type, time_us, data in event_bus.events
                if event_type == "packet_rx_succeeded" and int(data.get("packet_id", -1)) in packets
            }
            outcomes: list[TransmissionOutcome] = list(prefailed)
            delays: list[float] = []
            delivered_total = 0
            energy_total = 0.0
            for packet_id, (packet, intent, recipient, payload_bytes) in packets.items():
                success_event = success_events.get(packet_id)
                delivered = int(payload_bytes) if success_event is not None else 0
                delivered_total += delivered
                if success_event is not None:
                    delay_s = max(0.0, float(success_event[0] - packet.creation_time) / 1e6)
                    delays.append(delay_s)
                else:
                    delay_s = 0.0
                sender = int(intent.sender)
                energy_j = max(0.0, start_energy.get(sender, simulator.drones[sender].residual_energy) - simulator.drones[sender].residual_energy)
                energy_total += energy_j
                if int(intent.recipient) == GCS_NODE:
                    rate = float(snapshot.gcs_rates_bps[sender])
                    distance = float(np.linalg.norm(positions[sender] - gcs_position))
                else:
                    rate = float(snapshot.pair_rates_bps[int(intent.recipient), sender])
                    distance = float(np.linalg.norm(positions[sender] - positions[int(intent.recipient)]))
                outcomes.append(TransmissionOutcome(
                    sender=sender,
                    recipient=int(intent.recipient),
                    requested_bytes=int(intent.requested_bytes),
                    delivered_bytes=delivered,
                    rate_bps=rate,
                    distance_m=distance,
                    delay_s=delay_s,
                    tx_energy_j=energy_j,
                ))

            simulator.channel.close()

        return NetworkStepResult(
            outcomes=outcomes,
            attempted_bytes=attempted,
            delivered_bytes=int(delivered_total),
            byte_pdr=float(delivered_total / attempted) if attempted else 0.0,
            throughput_bps=float(delivered_total * 8.0 / dt_s),
            mean_delay_s=float(np.mean(delays)) if delays else 0.0,
            phy_failures=int(metrics.phy_failures) + len(prefailed),
            tx_energy_j=float(energy_total),
        )

def create_network_backend(name: str, cfg: dict[str, Any], seed: int = 0):
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized == "analytical":
        return AnalyticalNetworkBackend(cfg, seed=seed)
    if normalized in {"uavnetsim", "uav_net_sim"}:
        return UavNetSimBackend(cfg, seed=seed)
    raise ValueError(f"Unsupported network backend: {name}")
