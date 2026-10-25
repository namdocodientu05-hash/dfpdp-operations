"""
config.py
=========
Single source of truth for every tunable of the DFPDP system.

Legend used throughout the whole code base
------------------------------------------
[PAPER]                  : mechanism taken from Zhihui Sun et al. (2025), Neurocomputing 639, 130266
                           "A fast solution method for the Dynamic Flexible Pickup and Delivery Problem
                            with task allocation fairness for multiple vehicles"
[IMPLEMENTATION ASSUMPTION] : detail NOT fully specified by the paper; a concrete choice was made here
[REAL-TIME EXTENSION]    : feature added for real-time company deployment (not in the paper)

Every assumption / extension is switchable from this file so that
    BASE PAPER-LIKE STATE   vs.   REAL-TIME EXTENDED STATE
can be compared in controlled experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set


# --------------------------------------------------------------------------------------
# 1. TIME / CLOCK
# --------------------------------------------------------------------------------------
@dataclass
class ClockConfig:
    """[IMPLEMENTATION ASSUMPTION] one exact global event clock, in *minutes since midnight*.

    The paper does not describe clock synchronisation between ERP, TMS and the solver.
    We assume a single monotone global clock; every event timestamp is expressed in
    minutes-since-midnight (float).  Excel timestamps are converted on load.
    """
    day_start: float = 8 * 60.0          # 08:00
    day_end: float = 18 * 60.0           # 18:00
    time_unit_minutes: float = 1.0       # 1 unit == 1 minute
    # tie-break order when two events carry the same timestamp
    event_priority: Dict[str, int] = field(default_factory=lambda: {
        "SHIFT_START": 0,
        "VEHICLE_RECOVERY": 1,
        "TRAFFIC_UPDATE": 2,
        "GPS_UPDATE": 3,
        "PICKUP_COMPLETE": 4,
        "CUSTOMER_PICKUP_COMPLETE": 4,
        "DEPOT_LOAD_COMPLETE": 4,
        "SERVICE_COMPLETE": 4,
        "DELIVERY_COMPLETE": 5,
        "CUSTOMER_DELIVERY_COMPLETE": 5,
        "DEPOT_UNLOAD_COMPLETE": 5,
        "NEW_ORDER": 6,
        "CANCEL_ORDER": 7,
        "ORDER_PRIORITY_UPDATE": 8,
        "TIME_WINDOW_UPDATE": 9,
        "VEHICLE_BREAKDOWN": 10,
        "SHIFT_END": 11,
    })


# --------------------------------------------------------------------------------------
# 2. PROBLEM / PADDING SIZES  [PAPER: variable length sequences handled by masking]
# --------------------------------------------------------------------------------------
@dataclass
class ProblemConfig:
    M_max: int = 32                      # maximum order slots (network input width, never rebuilt)
    S_max: int = 4                       # maximum physical service locations per entity
    K_max: int = 8                       # maximum vehicles
    # [IMPLEMENTATION ASSUMPTION] what to do if more than M_max orders are simultaneously active
    #   "queue"  : the new order waits in a backlog until a slot frees (default, safe)
    #   "raise"  : raise an error (useful in unit tests)
    order_overflow_policy: str = "queue"


# --------------------------------------------------------------------------------------
# 3. STATE  (Z / U / D)  [PAPER §"state representation"]
# --------------------------------------------------------------------------------------
@dataclass
class StateConfig:
    # ---- baseline paper state -------------------------------------------------------
    # feature widths are FIXED so that the network is architecture-stable; extension
    # features are simply zeroed-out when their flag is False.  This is what makes the
    # BASE vs EXTENDED comparison a pure ablation.
    z_dim: int = 12     # GLOBAL: [z==0, z==1, z==2, released, is_dynamic, priority*, cancelled,
                        #          load_sign, next_is_depot, stage_progress, is_final_stage,
                        #          preloaded]                       (last 5: [FEDEX EXTENSION])
    u_dim: int = 10     # PER VEHICLE k: [w_norm, w/Q_k, remaining_cap_norm, capacity_feasible,
                        #                 load_after_norm, service_duration_norm, assigned_to_k,
                        #                 load_delta_norm, cargo_on_this_vehicle,
                        #                 ownership_locked]         (last 3: [FEDEX EXTENSION])
    d_dim: int = 8      # [min_dist, min_travel_time, waiting, arrival_norm, n_feas_loc, slack*, traffic*, same_loc]

    # [FEDEX EXTENSION] mixed-flow features are STRUCTURAL, not an ablation switch: an
    # OUTBOUND order simply cannot be represented without them.  They stay on in both the
    # BASE and the EXTENDED configuration so the state ablation keeps testing exactly the
    # real-time features it was designed to test.
    state_version: str = "fedex_mixed_flow_v1"   # checkpoints of another version are rejected

    # ---- REAL-TIME EXTENSION flags (OFF by default => paper baseline) ----------------
    use_realtime_traffic_state: bool = False   # R_t^k : predicted travel time / congestion factor
    use_time_window_slack: bool = False        # Slack_t^m = latest_feasible_time - estimated_arrival_time
    use_priority_feature: bool = False         # order priority inside Z
    use_vehicle_availability_flag: bool = True # cheap, kept on: masks broken vehicles out of the state

    # normalisation constants [IMPLEMENTATION ASSUMPTION] (paper does not give scaling)
    dist_scale: float = 100.0
    time_scale: float = 60.0
    weight_scale: float = 100.0


# --------------------------------------------------------------------------------------
# 4. TRAVEL TIME MODEL
# --------------------------------------------------------------------------------------
@dataclass
class TravelConfig:
    # "euclidean"  -> [PAPER] distance / speed
    # "traffic"    -> [REAL-TIME EXTENSION] distance / speed * congestion factor from TRAFFIC_UPDATE
    distance_mode: str = "euclidean"
    default_speed: float = 0.8            # distance units per minute
    use_speed_profile: bool = True        # per-hour multiplicative speed factor from the Speed_Profile sheet
    global_traffic_factor: float = 1.0    # live factor maintained by TrafficModel (1.0 = free flow)


# --------------------------------------------------------------------------------------
# 5. MCDSM  [PAPER: Minimum Cumulative-time Driven Selection Mechanism — vehicle selection]
# --------------------------------------------------------------------------------------
@dataclass
class MCDSMConfig:
    # k_t = argmin_{k in K_tilde} current_accumulated_time_k                       [PAPER]
    tie_break: str = "lowest_index"       # [IMPLEMENTATION ASSUMPTION] deterministic tie break
    first_step_initialisation: str = "paper_first_vehicle"
    #   "paper_first_vehicle" -> at t=0 all accumulated times are equal, select the first
    #                            feasible vehicle (index order)                    [PAPER-style]
    #   "random"              -> random feasible vehicle
    include_time_window_feasibility_in_mask: bool = True   # [REAL-TIME EXTENSION] pre-checks TWs
    log_every_selection: bool = True
    # [EXTENSION] exclusive assignment: once a dispatch decision for order m is ACCEPTED by
    # vehicle k, m is masked out for every other vehicle until it completes / is cancelled /
    # the shift ends / k breaks down.  This is a NEW CONSTRAINT on the formulation, not a
    # paper mechanism.  Offline (train / validate / evaluate / ablation) the commitment and
    # the execution happen in the same env.step(), so the flag is a provable no-op there.
    lock_committed_orders: bool = True


# --------------------------------------------------------------------------------------
# 6. SERVICE-LOCATION SELECTION  [PAPER: "flexible" = several physical locations per entity]
# --------------------------------------------------------------------------------------
@dataclass
class ServiceLocationConfig:
    # "paper_time_compatible" -> first location whose time window is compatible with the
    #                            vehicle's current time                            [PAPER-like]
    # "min_travel_plus_wait"  -> feasible location minimising travel + waiting     [REAL-TIME EXTENSION]
    selection_rule: str = "paper_time_compatible"
    # [IMPLEMENTATION ASSUMPTION] handling when *no* location has a feasible time window
    #   "allow_late_logged" : serve the least-late location, flag the order as LATE (default)
    #   "defer_order"       : mask the order for this vehicle/decision cycle, vehicle idles forward
    #   "drop_order"        : cancel the order and log it as infeasible
    infeasibility_rule: str = "allow_late_logged"


# --------------------------------------------------------------------------------------
# 7. REWARD  [PAPER: r_t = -(travel_time + waiting_time)]
# --------------------------------------------------------------------------------------
@dataclass
class RewardConfig:
    reward_scale: float = 0.01            # [IMPLEMENTATION ASSUMPTION] scaling only, sign/shape unchanged
    zero_movement_rule: bool = True       # [PAPER] two consecutive services at the same physical
                                          # location -> zero movement (travel cost 0)
    # ---- OPTIONAL REAL-TIME REWARD EXTENSIONS (all OFF by default) -------------------
    use_late_delivery_penalty: bool = False
    late_penalty_per_minute: float = 1.0
    use_priority_penalty: bool = False
    priority_penalty_weight: float = 1.0
    use_route_change_penalty: bool = False
    route_change_penalty: float = 2.0
    use_emergency_reassignment_penalty: bool = False
    emergency_reassignment_penalty: float = 20.0
    # ---- OPTIONAL fairness-in-reward experiment (OFF: fairness lives in MCDSM) -------
    use_fairness_in_reward: bool = False
    fairness_weight: float = 0.1


# --------------------------------------------------------------------------------------
# 8. NETWORK  (PESDVS = Parallel Encoders + Serial Decoder for Variable-length State)
# --------------------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    hidden_dim: int = 256                 # [PAPER]
    n_heads: int = 8                      # [PAPER]
    n_decoder_layers: int = 6             # [PAPER] six serial decoder layers
    ff_dim: int = 512                     # [IMPLEMENTATION ASSUMPTION]
    dropout: float = 0.0                  # [IMPLEMENTATION ASSUMPTION] paper does not mention dropout
    use_tanh_in_ff: bool = True           # [PAPER] "Tanh where consistent"
    critic_hidden: int = 128
    logit_clip: float = 10.0              # [IMPLEMENTATION ASSUMPTION] tanh-clipping of logits (stability)


# --------------------------------------------------------------------------------------
# 9. TRAINING  [PAPER: vanilla Actor-Critic with TD residual]
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    actor_lr: float = 1e-4                # [PAPER]
    critic_lr: float = 1e-3               # [PAPER]
    gamma: float = 0.01                   # [PAPER]
    grad_clip: float = 1.0                # [PAPER]
    n_episodes: int = 300
    validate_every: int = 20
    n_validation_episodes: int = 5
    seed: int = 42
    device: str = "cpu"
    entropy_coef: float = 0.0             # [IMPLEMENTATION ASSUMPTION] 0.0 keeps vanilla A-C exact
    checkpoint_dir: str = "checkpoints"
    best_checkpoint: str = "checkpoints/best_actor_critic.pt"
    last_checkpoint: str = "checkpoints/last_actor_critic.pt"
    log_dir: str = "outputs"
    online_learning_enabled: bool = False # separate experimental mode: never train during MODE B


# --------------------------------------------------------------------------------------
# 10. DATA
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    excel_path: str = "data/company_data_toy.xlsx"
    sheets: List[str] = field(default_factory=lambda: [
        "Vehicles", "Orders", "Service_Locations", "Speed_Profile", "Historical_Events",
        "Realtime_Order_Events", "Realtime_Vehicle_Events", "Realtime_Traffic", "Config",
    ])
    # chronological split (never random by row)  -> avoids leakage of future events
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    # synthetic augmentation preserves the empirical company distributions
    synthetic_augmentation: bool = True
    synthetic_episodes: int = 200
    data_source_tag: str = "REAL_HISTORICAL"     # or "SYNTHETIC_AUGMENTATION"
    n_orders_synthetic: Optional[int] = None     # None -> sample from empirical distribution
    dynamic_order_ratio: float = 0.4             # share of DYNAMIC orders in synthetic episodes


# --------------------------------------------------------------------------------------
# 11. REPLANNING TRIGGERS  [REAL-TIME EXTENSION]
# --------------------------------------------------------------------------------------
@dataclass
class ReplanConfig:
    replan_on_new_order: bool = True
    replan_on_cancel_order: bool = True
    replan_on_pickup_complete: bool = True
    replan_on_delivery_complete: bool = True
    replan_on_vehicle_breakdown: bool = True
    replan_on_vehicle_recovery: bool = True
    replan_on_time_window_update: bool = True
    replan_on_priority_update: bool = True
    replan_on_shift_start: bool = True
    # TRAFFIC_UPDATE: replan only if predicted travel time changes by >= rho
    traffic_rho: float = 0.15             # 15 % relative change
    # GPS_UPDATE: state update only, unless one of these thresholds fires
    gps_route_deviation_threshold: float = 5.0     # distance units off the expected position
    gps_lateness_threshold: float = 10.0           # predicted minutes late
    gps_travel_time_deterioration: float = 0.20    # 20 % worse than planned
    min_seconds_between_replans: float = 0.0       # rate limiting (0 = unlimited)
    periodic_replan_minutes: float = 30.0          # used by experiment B only


# --------------------------------------------------------------------------------------
# 12. BREAKDOWN HANDLING  [IMPLEMENTATION ASSUMPTION — paper is silent]
# --------------------------------------------------------------------------------------
@dataclass
class BreakdownConfig:
    #   "A_keep_until_recovery"      : cargo already picked up stays on the vehicle (DEFAULT BASELINE)
    #   "B_emergency_transshipment"  : picked-up orders are released back to z=1 and may be delivered
    #                                  by another vehicle (emergency reassignment penalty applies)
    #   "C_manual_intervention"      : orders are frozen and exported for human dispatch
    policy: str = "A_keep_until_recovery"
    allow_auto_reassign_picked_cargo: bool = False   # default: NOT allowed
    recovery_minutes: float = 60.0                   # used when no explicit RECOVERY event exists


# --------------------------------------------------------------------------------------
# 13. ONLINE INFERENCE
# --------------------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    decode_mode: str = "greedy"           # "greedy" | "sample" | "best_of_n"
    best_of_n: int = 8
    checkpoint: str = "checkpoints/best_actor_critic.pt"
    run_feasibility_validator: bool = True
    max_reselect_attempts: int = 5
    write_logs: bool = True
    output_dir: str = "outputs/online"
    # ---- [EXTENSION] academic closed-loop mode -------------------------------------
    # False (default, DEPLOYMENT): a recommendation is an *open-loop* proposal; the state
    #       only advances when the real ERP/TMS confirmation event arrives.
    # True  (ACADEMIC): every recommendation is assumed fully approved and is executed
    #       immediately through env.step(), so MODE B yields real experimental numbers
    #       (travel time, waiting time, fairness, completion) instead of latency only.
    auto_approve: bool = False
    auto_approve_max_cascade: int = 200    # safety bound on decisions per exogenous event


# --------------------------------------------------------------------------------------
# 14. MASTER CONFIG
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    clock: ClockConfig = field(default_factory=ClockConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    state: StateConfig = field(default_factory=StateConfig)
    travel: TravelConfig = field(default_factory=TravelConfig)
    mcdsm: MCDSMConfig = field(default_factory=MCDSMConfig)
    service_location: ServiceLocationConfig = field(default_factory=ServiceLocationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    breakdown: BreakdownConfig = field(default_factory=BreakdownConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # [IMPLEMENTATION ASSUMPTION] precedence: an explicitly pinned key (set by an experiment
    # preset or a CLI flag) always beats the workbook's Config sheet, so that an ablation
    # cannot be silently clobbered by the company file.
    pinned: Set[str] = field(default_factory=set)

    # ---------------------------------------------------------------- helpers ---------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["pinned"] = sorted(self.pinned)
        return d

    def pin(self, *keys: str) -> "Config":
        """Mark dotted keys as protected from workbook overrides."""
        self.pinned.update(keys)
        return self

    def set_pinned(self, key: str, value: object) -> "Config":
        """Set a dotted key and protect it from workbook overrides."""
        self.apply_overrides({key: value}, respect_pinned=False)
        return self.pin(key)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        cfg = Config()
        cfg.pinned = set(d.get("pinned", []))
        for section, values in d.items():
            if hasattr(cfg, section) and isinstance(values, dict):
                sub = getattr(cfg, section)
                for k, v in values.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        return cfg

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as fh:
            return Config.from_dict(json.load(fh))

    def apply_overrides(self, overrides: Dict[str, object],
                        respect_pinned: bool = True) -> "Config":
        """Apply dotted overrides, e.g. {"state.use_realtime_traffic_state": True}.

        Keys previously pinned (experiment presets, CLI flags) are skipped when
        ``respect_pinned`` is True.
        """
        for key, value in overrides.items():
            section, _, attr = key.partition(".")
            if not attr:
                raise KeyError(f"override must be 'section.attr', got {key}")
            if respect_pinned and key in self.pinned:
                continue
            sub = getattr(self, section)
            if not hasattr(sub, attr):
                raise KeyError(f"unknown config entry {key}")
            setattr(sub, attr, value)
        return self


# ---------------------------------------------------------------------------------------
# Ready-made experiment presets (used by evaluate.py)
# ---------------------------------------------------------------------------------------
def base_paper_config() -> Config:
    """BASE PAPER-LIKE STATE: Z/U/D only, Euclidean distance, MCDSM fairness, r=-(travel+wait)."""
    cfg = Config()
    cfg.set_pinned("state.use_realtime_traffic_state", False)
    cfg.set_pinned("state.use_time_window_slack", False)
    cfg.set_pinned("state.use_priority_feature", False)
    cfg.set_pinned("travel.distance_mode", "euclidean")
    cfg.set_pinned("service_location.selection_rule", "paper_time_compatible")
    return cfg


def realtime_extended_config() -> Config:
    """REAL-TIME EXTENDED STATE: traffic + slack + priority features, traffic travel times."""
    cfg = Config()
    cfg.set_pinned("state.use_realtime_traffic_state", True)
    cfg.set_pinned("state.use_time_window_slack", True)
    cfg.set_pinned("state.use_priority_feature", True)
    cfg.set_pinned("travel.distance_mode", "traffic")
    cfg.set_pinned("service_location.selection_rule", "min_travel_plus_wait")
    return cfg


def small_test_config() -> Config:
    """Tiny network / short episodes: unit tests and CI."""
    cfg = Config()
    cfg.network.hidden_dim = 32
    cfg.network.n_heads = 4
    cfg.network.n_decoder_layers = 2
    cfg.network.ff_dim = 64
    cfg.network.critic_hidden = 32
    cfg.problem.M_max = 8
    cfg.train.n_episodes = 3
    cfg.train.n_validation_episodes = 1
    return cfg
