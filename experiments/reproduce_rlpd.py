from generate import SbatchGenerator
from qam_matrix import (
    MAIN_DOMAINS,
    METHODS,
    SEEDS,
    TASKS,
    add_qam_run,
    env_name_for,
    get_data_root,
    validate_matrix,
)


run_group = "main-experiments"
num_jobs_per_gpu = 1
array_limit = 100


def generate(debug):
    gen = SbatchGenerator(j=num_jobs_per_gpu, limit=array_limit, prefix=("MUJOCO_GL=egl", "python main.py"))
    if debug:
        current_run_group = run_group + "_debug"
        gen.add_common_prefix(
            {
                "offline_steps": 100,
                "eval_episodes": 1,
                "eval_interval": 5,
                "start_training": 50,
                "online_steps": 100,
                "log_interval": 25,
            }
        )
    else:
        current_run_group = run_group
        gen.add_common_prefix(
            {
                "offline_steps": 1000000,
                "online_steps": 500000,
                "save_interval": 50000,
                "eval_interval": 50000,
            }
        )

    data_root = get_data_root()
    seeds = SEEDS[:1] if debug else SEEDS
    tasks = TASKS[:1] if debug else TASKS

    for seed in seeds:
        for domain in MAIN_DOMAINS:
            for task in tasks:
                env_name = env_name_for(domain, task)
                for method in METHODS:
                    add_qam_run(gen, current_run_group, method, seed, env_name, domain, data_root)

    return gen


validate_matrix()
print("# of methods:", len(METHODS))
print("methods:", list(METHODS))
print(
    "formal matrix:",
    f"{len(METHODS)} methods x {len(MAIN_DOMAINS)} domains x {len(TASKS)} tasks x {len(SEEDS)} seeds",
    "=",
    len(METHODS) * len(MAIN_DOMAINS) * len(TASKS) * len(SEEDS),
    "runs",
)
print(
    "debug matrix:",
    f"{len(METHODS)} methods x {len(MAIN_DOMAINS)} domains x 1 task x 1 seed",
    "=",
    len(METHODS) * len(MAIN_DOMAINS),
    "runs",
)

for debug in [True, False]:
    generator = generate(debug)
    sbatch_str_list = generator.generate_str()
    suffix = "_debug" if debug else ""
    for index, sbatch_str in enumerate(sbatch_str_list):
        with open(f"sbatch/{run_group}-part{index + 1}{suffix}.sh", "w") as f:
            f.write(sbatch_str)
