import os


METHODS = ("QAM", "QAM_FQL", "QAM_EDIT")

AGENT_FILES = {method: "agents/qam.py" for method in METHODS}

SEEDS = (
    10001,
    20002,
    30003,
    40004,
    50005,
)

MAIN_DOMAINS = (
    "cube-triple-play",
    "scene-play-sparse",
    "puzzle-3x3-play-sparse",
    "antmaze-large-navigate",
)

QUALITY_DOMAINS = (
    "humanoidmaze-medium-navigate",
    "cube-triple-play",
    "antmaze-large-navigate",
    "cube-double-play",
    "scene-play-sparse",
    "puzzle-3x3-play-sparse",
)

TASKS = (1, 2, 3, 4)
DEBUG_TASKS = (2,)

EXTERNAL_DATASET_DIRS = {
    "cube-quadruple-play": "cube-quadruple-play-100m-v0",
    "puzzle-4x4-play-sparse": "puzzle-4x4-play-100m-v0",
}

AGENT_PARAMS = dict(
    QAM={
        "puzzle-3x3-play-sparse": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "scene-play-sparse": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "cube-double-play": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "antmaze-large-navigate": dict(inv_temp=10.0, fql_alpha=0.0, edit_scale=0.0),
        "humanoidmaze-medium-navigate": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "cube-triple-play": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "cube-quadruple-play": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "antmaze-giant-navigate": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "humanoidmaze-large-navigate": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "puzzle-4x4-play-sparse": dict(inv_temp=30.0, fql_alpha=0.0, edit_scale=0.0),
    },
    QAM_FQL={
        "puzzle-3x3-play-sparse": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.0),
        "scene-play-sparse": dict(inv_temp=1.0, fql_alpha=300.0, edit_scale=0.0),
        "cube-double-play": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "antmaze-large-navigate": dict(inv_temp=3.0, fql_alpha=30.0, edit_scale=0.0),
        "humanoidmaze-medium-navigate": dict(inv_temp=1.0, fql_alpha=30.0, edit_scale=0.0),
        "cube-triple-play": dict(inv_temp=10.0, fql_alpha=300.0, edit_scale=0.0),
        "cube-quadruple-play": dict(inv_temp=0.3, fql_alpha=30.0, edit_scale=0.0),
        "puzzle-4x4-play-sparse": dict(inv_temp=3.0, fql_alpha=3.0, edit_scale=0.0),
        "antmaze-giant-navigate": dict(inv_temp=3.0, fql_alpha=30.0, edit_scale=0.0),
        "humanoidmaze-large-navigate": dict(inv_temp=0.3, fql_alpha=30.0, edit_scale=0.0),
    },
    QAM_EDIT={
        "puzzle-3x3-play-sparse": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.1),
        "scene-play-sparse": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "cube-double-play": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.0),
        "antmaze-large-navigate": dict(inv_temp=1.0, fql_alpha=0.0, edit_scale=0.1),
        "humanoidmaze-medium-navigate": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.1),
        "cube-triple-play": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.1),
        "cube-quadruple-play": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.1),
        "puzzle-4x4-play-sparse": dict(inv_temp=0.1, fql_alpha=0.0, edit_scale=0.9),
        "antmaze-giant-navigate": dict(inv_temp=10.0, fql_alpha=0.0, edit_scale=0.1),
        "humanoidmaze-large-navigate": dict(inv_temp=3.0, fql_alpha=0.0, edit_scale=0.1),
    },
)


def validate_matrix():
    for method in METHODS:
        domains = set(AGENT_PARAMS[method])
        missing = set(MAIN_DOMAINS) - domains
        if missing:
            raise ValueError(f"{method} is missing params for domains: {sorted(missing)}")
        for domain, params in AGENT_PARAMS[method].items():
            if params["fql_alpha"] * params["edit_scale"] != 0.0:
                raise ValueError(f"{method}/{domain} enables both FQL and EDIT mechanisms.")


def required_external_dataset_dirs(domains=MAIN_DOMAINS):
    return tuple(
        EXTERNAL_DATASET_DIRS[domain]
        for domain in domains
        if domain in EXTERNAL_DATASET_DIRS
    )


def formal_experiment_count():
    return len(METHODS) * len(MAIN_DOMAINS) * len(TASKS) * len(SEEDS)


def debug_experiment_count():
    return len(METHODS) * len(MAIN_DOMAINS) * len(DEBUG_TASKS)


def matrix_summary_lines():
    return (
        f"methods: {', '.join(METHODS)}",
        f"domains: {', '.join(MAIN_DOMAINS)}",
        f"tasks: {', '.join(str(task) for task in TASKS)}",
        f"debug tasks: {', '.join(str(task) for task in DEBUG_TASKS)}",
        f"seeds: {', '.join(str(seed) for seed in SEEDS)}",
        (
            f"formal matrix: {len(METHODS)} methods x {len(MAIN_DOMAINS)} domains "
            f"x {len(TASKS)} tasks x {len(SEEDS)} seeds = {formal_experiment_count()} runs"
        ),
        (
            f"debug matrix: {len(METHODS)} methods x {len(MAIN_DOMAINS)} domains "
            f"x {len(DEBUG_TASKS)} task x 1 seed = {debug_experiment_count()} runs"
        ),
    )


def get_data_root():
    data_root = os.environ.get("QAM_DATA_ROOT")
    required_dirs = required_external_dataset_dirs()
    if not data_root:
        if required_dirs:
            raise RuntimeError(
                "QAM_DATA_ROOT is required because the current matrix includes external 100M domains: "
                + ", ".join(required_dirs)
            )
        return ""
    return data_root.rstrip("/") + "/"


def env_name_for(domain, task, quality=False):
    name = domain[:-7] if domain.endswith("-sparse") else domain
    if quality:
        if "antmaze" in name or "humanoidmaze" in name:
            name = name.replace("navigate", "stitch")
        else:
            name = name.replace("play", "noisy")
    return f"{name}-singletask-task{task}-v0"


def horizon_length_for(domain):
    return 1 if "ant" in domain or "humanoid" in domain else 5


def base_kwargs(seed, env_name, domain, data_root):
    kwargs = {
        "seed": seed,
        "utd_ratio": 1,
        "agent.num_qs": 10,
        "env_name": env_name,
        "sparse": False if "sparse" not in domain else True,
        "horizon_length": horizon_length_for(domain),
        "agent.discount": 0.995 if "giant" in env_name or "humanoid" in env_name else 0.99,
        "agent.action_chunking": True,
        "agent.actor_hidden_dims": '"(512, 512, 512, 512)"',
        "agent.value_hidden_dims": '"(512, 512, 512, 512)"',
        "agent.batch_size": 256,
        "agent.rho": 0.0 if "humanoid" in domain else 0.5,
    }
    if domain in EXTERNAL_DATASET_DIRS:
        if not data_root:
            raise RuntimeError(f"QAM_DATA_ROOT is required for external 100M domain: {domain}")
        kwargs["ogbench_dataset_dir"] = data_root + EXTERNAL_DATASET_DIRS[domain] + "/"
    return kwargs


def add_qam_run(gen, run_group, method, seed, env_name, domain, data_root):
    kwargs = {
        "run_group": f"{run_group}-{method}",
        "agent": AGENT_FILES[method],
        "tags": method,
        **base_kwargs(seed, env_name, domain, data_root),
    }
    for key, value in AGENT_PARAMS[method][domain].items():
        kwargs[f"agent.{key}"] = value
    gen.add_run(kwargs)
