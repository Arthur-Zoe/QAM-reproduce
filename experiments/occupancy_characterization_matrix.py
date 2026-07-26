"""Pure experiment matrix for transition-occupancy characterization."""

from dataclasses import dataclass
import math

try:
    from .qam_matrix import (
        AGENT_FILES,
        AGENT_PARAMS,
        base_kwargs,
        env_name_for,
    )
except ImportError:
    from qam_matrix import (
        AGENT_FILES,
        AGENT_PARAMS,
        base_kwargs,
        env_name_for,
    )


BASE_COMMIT = "9a2e08465342ac9d3413348c657a94d8351ad39f"
METHOD = "QAM_FQL"
DOMAIN = "cube-triple-play"
TASK = 2
ENV_NAME = env_name_for(DOMAIN, TASK)
SAVE_ROOT = "exp/qam-occupancy-characterization"
LOCAL_SEEDS = (10001,)
PILOT_SEEDS = (10001, 20002, 30003)


@dataclass(frozen=True)
class DynamicsCondition:
    """Immutable action-dynamics condition."""

    name: str
    condition_family: str
    action_gain: float
    action_delay: int

    @property
    def severity(self):
        if self.condition_family == "nominal":
            return 0.0
        if self.condition_family == "gain":
            return abs(1.0 - self.action_gain)
        if self.condition_family == "delay":
            return float(self.action_delay)
        raise ValueError(
            f"unsupported condition family: {self.condition_family!r}"
        )


LOCAL_CONDITIONS = (
    DynamicsCondition("nominal", "nominal", 1.0, 0),
    DynamicsCondition("gain_0p7", "gain", 0.7, 0),
)
PILOT_CONDITIONS = (
    DynamicsCondition("nominal", "nominal", 1.0, 0),
    DynamicsCondition("gain_0p9", "gain", 0.9, 0),
    DynamicsCondition("gain_0p7", "gain", 0.7, 0),
    DynamicsCondition("gain_0p5", "gain", 0.5, 0),
    DynamicsCondition("delay_1", "delay", 1.0, 1),
    DynamicsCondition("delay_2", "delay", 1.0, 2),
    DynamicsCondition("delay_3", "delay", 1.0, 3),
)

LOCAL_PROTOCOL = {
    "offline_steps": 5000,
    "online_steps": 5000,
    "start_training": 5001,
    "log_interval": 1000,
    "eval_interval": 1000,
    "eval_episodes": 3,
    "video_episodes": 0,
    "save_interval": 0,
    "online_save_interval": 0,
    "auto_cleanup": False,
    "balanced_sampling": False,
    "recent_dynamics_capacity": 1000,
    "occupancy_detector": True,
    "occupancy_hidden_dim": 256,
    "occupancy_num_hidden_layers": 2,
    "occupancy_lr": 3e-4,
    "occupancy_batch_size": 128,
    "occupancy_start_size": 500,
    "occupancy_update_interval": 500,
    "occupancy_updates_per_interval": 5,
}
PILOT_PROTOCOL = {
    "offline_steps": 100000,
    "online_steps": 50000,
    "start_training": 50001,
    "log_interval": 10000,
    "eval_interval": 10000,
    "eval_episodes": 10,
    "video_episodes": 0,
    "save_interval": 0,
    "online_save_interval": 10000,
    "auto_cleanup": False,
    "balanced_sampling": False,
    "recent_dynamics_capacity": 5000,
    "occupancy_detector": True,
    "occupancy_hidden_dim": 256,
    "occupancy_num_hidden_layers": 2,
    "occupancy_lr": 3e-4,
    "occupancy_batch_size": 256,
    "occupancy_start_size": 1000,
    "occupancy_update_interval": 1000,
    "occupancy_updates_per_interval": 20,
}


def infer_agent_online_updates(kwargs):
    """Infer whether the online loop can update the QAM agent."""
    return (
        "enabled"
        if kwargs["start_training"] <= kwargs["online_steps"]
        else "disabled"
    )


def infer_behavior_policy(kwargs):
    """Infer the policy that supplies online characterization actions."""
    balanced_sampling = kwargs["balanced_sampling"]
    start_training = kwargs["start_training"]
    online_steps = kwargs["online_steps"]
    if balanced_sampling:
        if start_training > online_steps:
            return "random_only"
        if start_training > 1:
            return "random_warmup_then_agent"
        return "online_agent"
    if start_training > online_steps:
        return "fixed_offline_agent"
    return "online_agent"


def validate_condition_definitions(conditions):
    """Validate condition names and mutually exclusive family semantics."""
    names = [condition.name for condition in conditions]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique.")
    nominal_count = 0
    for condition in conditions:
        if not condition.name:
            raise ValueError("condition name must be non-empty.")
        if (
            not isinstance(condition.action_gain, (int, float))
            or isinstance(condition.action_gain, bool)
            or not math.isfinite(float(condition.action_gain))
            or condition.action_gain <= 0
        ):
            raise ValueError(
                f"{condition.name}: action_gain must be finite and positive."
            )
        if (
            isinstance(condition.action_delay, bool)
            or not isinstance(condition.action_delay, int)
            or condition.action_delay < 0
        ):
            raise ValueError(
                f"{condition.name}: action_delay must be a non-negative integer."
            )
        family = condition.condition_family
        if family == "nominal":
            nominal_count += 1
            if (
                condition.name != "nominal"
                or condition.action_gain != 1.0
                or condition.action_delay != 0
            ):
                raise ValueError(
                    "nominal must use gain=1.0 and delay=0."
                )
        elif family == "gain":
            if condition.action_delay != 0:
                raise ValueError(
                    f"{condition.name}: gain conditions require delay=0."
                )
        elif family == "delay":
            if condition.action_gain != 1.0:
                raise ValueError(
                    f"{condition.name}: delay conditions require gain=1.0."
                )
        else:
            raise ValueError(
                f"{condition.name}: unsupported family {family!r}."
            )
    if nominal_count != 1:
        raise ValueError("condition matrix must contain exactly one nominal.")


def validate_run_records(runs, expected_count=None):
    """Validate run identity uniqueness and optional matrix cardinality."""
    identities = [(run["condition"], run["seed"]) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate (condition, seed) run.")
    if expected_count is not None and len(runs) != expected_count:
        raise ValueError(
            f"expected {expected_count} runs, got {len(runs)}."
        )
    for run in runs:
        if "kwargs" not in run:
            continue
        inferred_policy = infer_behavior_policy(run["kwargs"])
        inferred_updates = infer_agent_online_updates(run["kwargs"])
        if (
            inferred_policy != "fixed_offline_agent"
            or inferred_updates != "disabled"
        ):
            raise ValueError(
                "characterization requires behavior_policy="
                "fixed_offline_agent and agent_online_updates=disabled; "
                f"inferred {inferred_policy!r} and {inferred_updates!r}."
            )
        if run.get("behavior_policy") != inferred_policy:
            raise ValueError(
                "run behavior_policy metadata does not match its flags."
            )
        if run.get("agent_online_updates") != inferred_updates:
            raise ValueError(
                "run agent_online_updates metadata does not match its flags."
            )


def validate_characterization_matrix(
    local_conditions=LOCAL_CONDITIONS,
    pilot_conditions=PILOT_CONDITIONS,
    local_seeds=LOCAL_SEEDS,
    pilot_seeds=PILOT_SEEDS,
):
    """Validate both public matrix modes and their required sizes."""
    validate_condition_definitions(local_conditions)
    validate_condition_definitions(pilot_conditions)
    if len(local_conditions) * len(local_seeds) != 2:
        raise ValueError("local characterization matrix must contain 2 runs.")
    if len(pilot_conditions) * len(pilot_seeds) != 21:
        raise ValueError("pilot characterization matrix must contain 21 runs.")


def _normalise_matrix_value(value):
    if (
        isinstance(value, str)
        and len(value) >= 2
        and value[0] == value[-1] == '"'
    ):
        return value[1:-1]
    return value


def _run_kwargs(condition, seed, run_group, protocol):
    kwargs = {
        "run_group": run_group,
        "agent": AGENT_FILES[METHOD],
        "tags": (
            f"{METHOD},occupancy-characterization,{condition.name}"
        ),
        "save_dir": SAVE_ROOT,
        **{
            key: _normalise_matrix_value(value)
            for key, value in base_kwargs(
                seed,
                ENV_NAME,
                DOMAIN,
                data_root="",
            ).items()
        },
        **protocol,
        "train_action_gain": condition.action_gain,
        "train_action_delay": condition.action_delay,
        "eval_action_gain": condition.action_gain,
        "eval_action_delay": condition.action_delay,
    }
    for key, value in AGENT_PARAMS[METHOD][DOMAIN].items():
        kwargs[f"agent.{key}"] = value
    return kwargs


def _run_record(condition, seed, run_group, protocol):
    kwargs = _run_kwargs(condition, seed, run_group, protocol)
    return {
        "condition": condition.name,
        "condition_family": condition.condition_family,
        "severity": condition.severity,
        "seed": seed,
        "method": METHOD,
        "domain": DOMAIN,
        "task": TASK,
        "env_name": ENV_NAME,
        "run_group": run_group,
        "tags": (
            f"{METHOD},occupancy-characterization,{condition.name}"
        ),
        "behavior_policy": infer_behavior_policy(kwargs),
        "agent_online_updates": infer_agent_online_updates(kwargs),
        "kwargs": kwargs,
    }


def build_characterization_runs(mode, run_group):
    """Build deterministic run records for local or 21-run pilot mode."""
    if not isinstance(run_group, str) or not run_group:
        raise ValueError("run_group must be a non-empty string.")
    validate_characterization_matrix()
    if mode == "local":
        conditions = LOCAL_CONDITIONS
        seeds = LOCAL_SEEDS
        protocol = LOCAL_PROTOCOL
        expected_count = 2
    elif mode == "pilot":
        conditions = PILOT_CONDITIONS
        seeds = PILOT_SEEDS
        protocol = PILOT_PROTOCOL
        expected_count = 21
    else:
        raise ValueError(f"unsupported characterization mode: {mode!r}")

    runs = tuple(
        _run_record(condition, seed, run_group, protocol)
        for condition in conditions
        for seed in seeds
    )
    validate_run_records(runs, expected_count=expected_count)
    return runs
