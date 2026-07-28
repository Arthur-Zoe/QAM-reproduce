"""Synthetic GPU smoke test for the standalone dynamics-shift bridge."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dynamics_shift_bridge import (
    DynamicsShiftBridge,
    DynamicsShiftBridgeConfig,
)


def train_linear_case(gain):
    generator = np.random.default_rng(123)
    matrix = np.array(
        [[0.7, -0.2], [0.1, 0.5], [-0.3, 0.4]],
        dtype=np.float32,
    )
    observations = generator.uniform(
        -0.5, 0.5, (256, 3)
    ).astype(np.float32)
    actions = generator.uniform(
        -0.7, 0.7, (256, 2)
    ).astype(np.float32)
    offline_batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": observations + actions @ matrix.T,
    }
    bridge = DynamicsShiftBridge.create(
        seed=5,
        example_observations=observations,
        example_actions=actions,
        config=DynamicsShiftBridgeConfig(
            hidden_dim=32,
            num_hidden_layers=2,
            learning_rate=3e-3,
            correction_steps=40,
            correction_step_size=0.2,
            action_l2_weight=1e-3,
            max_residual=0.5,
        ),
    )
    for _ in range(500):
        bridge, _ = bridge.update_offline(offline_batch)
    bridge = bridge.synchronize_online_from_offline()
    online_batch = {
        **offline_batch,
        "next_observations": (
            observations + (gain * actions) @ matrix.T
        ),
    }
    for _ in range(300):
        bridge, _ = bridge.update_online(online_batch)
    held_observations = generator.uniform(
        -0.5, 0.5, (64, 3)
    ).astype(np.float32)
    base_actions = (
        generator.uniform(-0.7, 0.7, (64, 2)).astype(np.float32)
        * 0.6
    )
    corrected_actions, metrics = bridge.correct_actions(
        held_observations, base_actions
    )
    return (
        bridge,
        held_observations,
        base_actions,
        np.asarray(corrected_actions),
        metrics,
    )


def train_nonlinear_case():
    generator = np.random.default_rng(321)
    state_matrix = np.array(
        [
            [0.4, -0.2, 0.1],
            [0.1, 0.3, -0.4],
            [-0.2, 0.2, 0.5],
        ],
        dtype=np.float32,
    )
    action_matrix = np.array(
        [[0.6, -0.1], [0.2, 0.5], [-0.4, 0.3]],
        dtype=np.float32,
    )

    def delta(observations, actions, gain):
        return np.tanh(
            observations @ state_matrix.T
            + (gain * actions) @ action_matrix.T
        ).astype(np.float32)

    observations = generator.uniform(
        -0.5, 0.5, (512, 3)
    ).astype(np.float32)
    actions = generator.uniform(
        -0.7, 0.7, (512, 2)
    ).astype(np.float32)
    offline_batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": (
            observations + delta(observations, actions, 1.0)
        ),
    }
    bridge = DynamicsShiftBridge.create(
        seed=21,
        example_observations=observations,
        example_actions=actions,
        config=DynamicsShiftBridgeConfig(
            hidden_dim=64,
            num_hidden_layers=2,
            learning_rate=3e-3,
            correction_steps=60,
            correction_step_size=0.15,
            action_l2_weight=1e-3,
            max_residual=0.5,
        ),
    )
    for _ in range(700):
        bridge, _ = bridge.update_offline(offline_batch)
    bridge = bridge.synchronize_online_from_offline()
    online_batch = {
        **offline_batch,
        "next_observations": (
            observations + delta(observations, actions, 0.7)
        ),
    }
    for _ in range(500):
        bridge, _ = bridge.update_online(online_batch)
    held_observations = generator.uniform(
        -0.5, 0.5, (128, 3)
    ).astype(np.float32)
    base_actions = generator.uniform(
        -0.4, 0.4, (128, 2)
    ).astype(np.float32)
    _, metrics = bridge.correct_actions(
        held_observations, base_actions
    )
    return metrics


def main():
    nominal = train_linear_case(1.0)
    gain_0p7 = train_linear_case(0.7)
    gain_1p3 = train_linear_case(1.3)
    (
        gain_bridge,
        held_observations,
        base_actions,
        gain_corrected_actions,
        gain_metrics,
    ) = gain_0p7

    near_bound_actions = np.where(
        base_actions >= 0.0, 0.995, -0.995
    ).astype(np.float32)
    near_bound_corrected, near_bound_metrics = (
        gain_bridge.correct_actions(
            held_observations, near_bound_actions
        )
    )
    limited_bridge = gain_bridge.replace(
        max_residual=jnp.full_like(gain_bridge.max_residual, 0.02)
    )
    limited_corrected, limited_metrics = limited_bridge.correct_actions(
        held_observations, base_actions
    )
    results = {
        "nominal": nominal[-1],
        "gain_0p7": gain_metrics,
        "gain_1p3": gain_1p3[-1],
        "nonlinear_gain_0p7": train_nonlinear_case(),
        "near_action_bounds": near_bound_metrics,
        "limited_residual": limited_metrics,
    }

    def ratio(metrics):
        return float(metrics["post_match_mse"]) / max(
            float(metrics["pre_match_mse"]),
            np.finfo(np.float32).tiny,
        )

    assert ratio(results["nominal"]) <= 1.0 + 1e-5
    assert float(results["nominal"]["residual_l2_mean"]) < 5e-2
    assert ratio(gain_metrics) <= 0.5
    assert np.mean(np.linalg.norm(gain_corrected_actions, axis=-1)) > (
        np.mean(np.linalg.norm(base_actions, axis=-1))
    )
    assert ratio(results["gain_1p3"]) < 1.0
    assert np.mean(np.linalg.norm(gain_1p3[3], axis=-1)) < np.mean(
        np.linalg.norm(gain_1p3[2], axis=-1)
    )
    assert ratio(results["nonlinear_gain_0p7"]) < 1.0
    assert ratio(near_bound_metrics) <= 1.0 + 1e-5
    assert float(near_bound_metrics["action_clip_fraction"]) > 0.0
    assert ratio(limited_metrics) <= 1.0 + 1e-5
    assert float(limited_metrics["residual_clip_fraction"]) > 0.0

    for actions, actions_base, bridge in (
        (near_bound_corrected, near_bound_actions, gain_bridge),
        (limited_corrected, base_actions, limited_bridge),
    ):
        assert np.all(actions >= np.asarray(bridge.action_low))
        assert np.all(actions <= np.asarray(bridge.action_high))
        assert np.all(
            np.abs(actions - actions_base)
            <= np.asarray(bridge.max_residual) + 1e-6
        )

    print(f"backend: {jax.default_backend()}")
    print(f"device: {jax.devices()[0]}")
    all_values = []
    for name, metrics in results.items():
        pre_match_mse = float(metrics["pre_match_mse"])
        post_match_mse = float(metrics["post_match_mse"])
        residual_l2_mean = float(metrics["residual_l2_mean"])
        metric_values = [
            float(np.asarray(value)) for value in metrics.values()
        ]
        all_values.extend(metric_values)
        print(
            f"{name} pre/post MSE: "
            f"{pre_match_mse:.9g} / {post_match_mse:.9g} "
            f"(ratio {ratio(metrics):.6f})"
        )
        print(
            f"{name} mean residual norm: "
            f"{residual_l2_mean:.9g}"
        )
        print(
            f"{name} action/residual clip fractions: "
            f"{float(metrics['action_clip_fraction']):.6f} / "
            f"{float(metrics['residual_clip_fraction']):.6f}"
        )
    print(f"all values finite: {bool(np.all(np.isfinite(all_values)))}")
    assert np.all(np.isfinite(all_values))


if __name__ == "__main__":
    main()
