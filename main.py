import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.dynamics_shift import apply_dynamics_shift

from utils.flax_utils import save_agent, restore_agent
from utils.datasets import Dataset, ReplayBuffer
from utils.online_checkpoint import (
    CHECKPOINT_FILENAME,
    load_online_checkpoint,
    online_start_step,
    read_progress,
    save_online_checkpoint,
    should_save_online_checkpoint,
)
from utils.recent_dynamics_buffer import (
    RecentDynamicsBuffer,
    create_recent_transition_template,
)
from utils.dynamics_shift_bridge_runtime import (
    DynamicsShiftBridgeRuntime,
    DynamicsShiftBridgeRuntimeConfig,
    bridge_step_environment,
    call_preserving_global_numpy_rng,
    extract_primitive_transitions,
    validate_bridge_checkpoint_resume,
    validate_bridge_runtime_config,
)
from utils.transition_occupancy import (
    TRANSITION_FIELDS,
    TransitionOccupancyDetector,
    average_occupancy_metrics,
    sample_occupancy_transition_batches,
    should_update_occupancy_detector,
)

from evaluation import evaluate
from agents import agents
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('tags', 'Default', 'Wandb tag.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 500000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 1000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 50000, 'Logging interval.') #每多少步记录一次日志
flags.DEFINE_integer('eval_interval', 50000, 'Evaluation interval.')    #每多少步评测一次
flags.DEFINE_integer('save_interval', 50000, 'Save interval.') # for the offline stage only.
flags.DEFINE_integer('online_save_interval', 0, 'Online checkpoint interval; saves at the next episode boundary.')
flags.DEFINE_integer(
    'recent_dynamics_capacity',
    0,
    'Capacity of the recent online transition buffer; 0 disables it.',
)
flags.DEFINE_bool(
    'dynamics_bridge',
    False,
    'Enable the shadow-only Dynamics-Shift Bridge lifecycle.',
)
flags.DEFINE_bool(
    'dynamics_bridge_apply_correction',
    False,
    'Execute gated and ramped primitive Bridge corrections.',
)
flags.DEFINE_integer('bridge_hidden_dim', 256, 'Bridge MLP hidden width.')
flags.DEFINE_integer(
    'bridge_num_hidden_layers', 2, 'Bridge MLP hidden-layer count.'
)
flags.DEFINE_float('bridge_lr', 3e-4, 'Bridge learning rate.')
flags.DEFINE_float(
    'bridge_clip_grad_norm', 10.0, 'Bridge gradient clipping norm.'
)
flags.DEFINE_integer(
    'bridge_offline_steps', 10000, 'Bridge offline pretraining updates.'
)
flags.DEFINE_integer(
    'bridge_offline_batch_size', 256, 'Bridge offline batch size.'
)
flags.DEFINE_integer(
    'bridge_online_start_size',
    500,
    'Recent-transition count required before Bridge online updates.',
)
flags.DEFINE_integer(
    'bridge_online_update_interval',
    500,
    'Online environment-step interval between Bridge update bursts.',
)
flags.DEFINE_integer(
    'bridge_online_updates_per_interval',
    20,
    'Bridge online-model updates per scheduled burst.',
)
flags.DEFINE_integer(
    'bridge_online_batch_size', 256, 'Bridge online batch size.'
)
flags.DEFINE_integer(
    'bridge_correction_steps', 10, 'Shadow action-correction steps.'
)
flags.DEFINE_float(
    'bridge_correction_step_size', 0.1, 'Shadow correction step size.'
)
flags.DEFINE_float(
    'bridge_dynamics_match_weight',
    1.0,
    'Normalized dynamics-matching objective weight.',
)
flags.DEFINE_float(
    'bridge_action_l2_weight',
    0.01,
    'Raw-action residual regularization weight.',
)
flags.DEFINE_float(
    'bridge_max_residual', 0.1, 'Per-component raw-action residual bound.'
)
flags.DEFINE_float(
    'bridge_gate_max_online_eval_mse',
    0.10,
    'Maximum normalized held-out online-model MSE for execution.',
)
flags.DEFINE_float(
    'bridge_gate_uncertainty_multiplier',
    1.0,
    'Online-model uncertainty multiplier in the shift-excess gate.',
)
flags.DEFINE_float(
    'bridge_gate_min_shift_excess',
    0.005,
    'Minimum normalized shift excess required for execution.',
)
flags.DEFINE_float(
    'bridge_gate_min_relative_improvement',
    0.20,
    'Minimum relative correction improvement required for execution.',
)
flags.DEFINE_integer(
    'bridge_apply_ramp_steps',
    1000,
    'Gate-open steps used to ramp the executed residual to full scale.',
)
flags.DEFINE_float(
    'bridge_apply_residual_scale',
    1.0,
    'Maximum fraction of the bounded candidate residual to execute.',
)
flags.DEFINE_integer(
    'bridge_normalization_max_samples',
    100000,
    'Maximum primitive offline transitions used for normalization.',
)
flags.DEFINE_bool(
    'occupancy_detector',
    False,
    'Enable the transition occupancy shift detector.',
)
flags.DEFINE_integer(
    'occupancy_hidden_dim',
    256,
    'Hidden width of the transition occupancy detector.',
)
flags.DEFINE_integer(
    'occupancy_num_hidden_layers',
    2,
    'Number of hidden layers in the transition occupancy detector.',
)
flags.DEFINE_float(
    'occupancy_lr',
    3e-4,
    'Learning rate of the transition occupancy detector.',
)
flags.DEFINE_integer(
    'occupancy_batch_size',
    256,
    'Balanced offline and recent-online batch size per detector update.',
)
flags.DEFINE_integer(
    'occupancy_start_size',
    1000,
    'Minimum recent online transitions before detector updates.',
)
flags.DEFINE_integer(
    'occupancy_update_interval',
    1000,
    'Online environment-step interval between detector update bursts.',
)
flags.DEFINE_integer(
    'occupancy_updates_per_interval',
    20,
    'Number of detector gradient updates per update burst.',
)
flags.DEFINE_integer('start_training', 5000, 'when does training start')    # 对于在线阶段，前多少步只收集数据不训练

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/qam.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_float('train_action_gain', 1.0, 'Action gain for online training interaction.')
flags.DEFINE_integer('train_action_delay', 0, 'Action delay steps for online training interaction.')
flags.DEFINE_float('eval_action_gain', 1.0, 'Action gain for evaluation interaction.')
flags.DEFINE_integer('eval_action_delay', 0, 'Action delay steps for evaluation interaction.')

flags.DEFINE_bool('auto_cleanup', True, "remove all intermediate checkpoints when the run finishes")

flags.DEFINE_bool('balanced_sampling', False, "sample half offline and online replay buffer")

OCCUPANCY_CLIP_GRAD_NORM = 10.0
OCCUPANCY_SEED_OFFSET = 1_000_003

def save_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        csv_logger.save(os.path.join(save_dir, f"{prefix}_sv.csv"))

def restore_csv_loggers(csv_loggers, save_dir):
    for prefix, csv_logger in csv_loggers.items():
        if os.path.exists(os.path.join(save_dir, f"{prefix}_sv.csv")):
            csv_logger.restore(os.path.join(save_dir, f"{prefix}_sv.csv"))

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    bridge_runtime_config = DynamicsShiftBridgeRuntimeConfig(
        enabled=FLAGS.dynamics_bridge,
        apply_correction=FLAGS.dynamics_bridge_apply_correction,
        hidden_dim=FLAGS.bridge_hidden_dim,
        num_hidden_layers=FLAGS.bridge_num_hidden_layers,
        learning_rate=FLAGS.bridge_lr,
        clip_grad_norm=FLAGS.bridge_clip_grad_norm,
        offline_steps=FLAGS.bridge_offline_steps,
        offline_batch_size=FLAGS.bridge_offline_batch_size,
        online_start_size=FLAGS.bridge_online_start_size,
        online_update_interval=FLAGS.bridge_online_update_interval,
        online_updates_per_interval=(
            FLAGS.bridge_online_updates_per_interval
        ),
        online_batch_size=FLAGS.bridge_online_batch_size,
        correction_steps=FLAGS.bridge_correction_steps,
        correction_step_size=FLAGS.bridge_correction_step_size,
        dynamics_match_weight=FLAGS.bridge_dynamics_match_weight,
        action_l2_weight=FLAGS.bridge_action_l2_weight,
        max_residual=FLAGS.bridge_max_residual,
        gate_max_online_eval_mse=(
            FLAGS.bridge_gate_max_online_eval_mse
        ),
        gate_uncertainty_multiplier=(
            FLAGS.bridge_gate_uncertainty_multiplier
        ),
        gate_min_shift_excess=(
            FLAGS.bridge_gate_min_shift_excess
        ),
        gate_min_relative_improvement=(
            FLAGS.bridge_gate_min_relative_improvement
        ),
        apply_ramp_steps=FLAGS.bridge_apply_ramp_steps,
        apply_residual_scale=FLAGS.bridge_apply_residual_scale,
        normalization_max_samples=(
            FLAGS.bridge_normalization_max_samples
        ),
    )
    validate_bridge_runtime_config(
        bridge_runtime_config,
        recent_dynamics_capacity=FLAGS.recent_dynamics_capacity,
        online_save_interval=FLAGS.online_save_interval,
        log_interval=FLAGS.log_interval,
    )
    if FLAGS.online_save_interval < 0:
        raise ValueError(
            "online_save_interval must be non-negative; "
            f"got {FLAGS.online_save_interval}."
        )
    if FLAGS.recent_dynamics_capacity < 0:
        raise ValueError(
            "recent_dynamics_capacity must be non-negative; "
            f"got {FLAGS.recent_dynamics_capacity}."
        )
    if FLAGS.occupancy_detector:
        if FLAGS.recent_dynamics_capacity <= 0:
            raise ValueError(
                "occupancy_detector requires "
                "recent_dynamics_capacity > 0."
            )
        for name in (
            "occupancy_hidden_dim",
            "occupancy_num_hidden_layers",
            "occupancy_batch_size",
            "occupancy_start_size",
            "occupancy_update_interval",
            "occupancy_updates_per_interval",
        ):
            value = getattr(FLAGS, name)
            if value <= 0:
                raise ValueError(
                    f"{name} must be positive when occupancy_detector is "
                    f"enabled; got {value}."
                )
        if (
            not np.isfinite(FLAGS.occupancy_lr)
            or FLAGS.occupancy_lr <= 0.0
        ):
            raise ValueError(
                "occupancy_lr must be finite and positive when "
                f"occupancy_detector is enabled; got {FLAGS.occupancy_lr}."
            )
        if (
            FLAGS.occupancy_start_size
            > FLAGS.recent_dynamics_capacity
        ):
            raise ValueError(
                "occupancy_start_size must not exceed "
                "recent_dynamics_capacity; "
                f"got {FLAGS.occupancy_start_size} > "
                f"{FLAGS.recent_dynamics_capacity}."
            )
    if (
        FLAGS.ogbench_dataset_dir is not None
        and FLAGS.online_save_interval > 0
    ):
        raise NotImplementedError(
            "Online checkpointing with a custom ogbench_dataset_dir is not "
            "supported because custom dataset rotation recovery is not implemented."
        )

    exp_name = get_exp_name(FLAGS)
    run = setup_wandb(
        project=os.environ.get("WANDB_PROJECT", "qam-reproduce"),
        entity=os.environ.get("WANDB_ENTITY", None),
        group=FLAGS.run_group,
        name=exp_name,
        tags=FLAGS.tags.split(","),
        mode=os.environ.get("WANDB_MODE", "online"),
    )
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    if (
        bridge_runtime_config.enabled
        and os.path.isdir(FLAGS.save_dir)
        and not os.path.exists(
            os.path.join(FLAGS.save_dir, "token.tk")
        )
    ):
        preflight_load_stage, _ = read_progress(FLAGS.save_dir)
        validate_bridge_checkpoint_resume(
            True, preflight_load_stage
        )
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        # assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]

        if FLAGS.dataset_proportion < 1.:
            num_datasets = len(dataset_paths)
            num_subset_datasets = max(1, int(num_datasets * FLAGS.dataset_proportion))
            print("actual data proportion:", num_subset_datasets / num_datasets)
            dataset_paths = dataset_paths[:num_subset_datasets]

        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # Keep the environment used for later dataset relabeling free of online shifts.
    dataset_env = env
    env = apply_dynamics_shift(
        env,
        action_gain=FLAGS.train_action_gain,
        action_delay=FLAGS.train_action_delay,
    )
    eval_env = apply_dynamics_shift(
        eval_env,
        action_gain=FLAGS.eval_action_gain,
        action_delay=FLAGS.eval_action_delay,
    )

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    
    config = FLAGS.agent
    discount = FLAGS.agent.discount
    config["horizon_length"] = FLAGS.horizon_length

    # 处理数据集，主要是处理数据量和稀疏奖励，以及转换成action chunk的格式
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        action_dim = ds["actions"].shape[-1]
        # 这里直接把动作chunk化了，后续每次sample的时候就不需要再处理了
        if FLAGS.dataset_proportion < 1.0:
            #  如果数据集过大，无法一次性加载到内存中，可以通过调整dataset_proportion和dataset_replace_interval来分批加载数据集的一部分进行训练
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        # 这里的动作chunk化是指把连续horizon_length个动作合并成一个chunk，作为新的动作输入到模型中
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()} #先把原始数据集转换成字典，方便修改
            ds_dict["rewards"] = sparse_rewards #制作稀疏奖励
            ds = Dataset.create(**ds_dict)  #把动作chunk化

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    bridge_primitive_transitions = None
    if bridge_runtime_config.enabled:
        # Freeze raw primitive transition views before the first QAM
        # sequence sample, any action-chunk construction, or later dataset
        # replacement/relabeling.
        bridge_primitive_transitions = extract_primitive_transitions(
            train_dataset,
            tuple(env.action_space.shape),
        )
    # 从数据集中取一个样例 batch
    example_batch = train_dataset.sample(())
    
    # 根据 config['agent_name'] 找到对应算法类
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,  # 随机种子,作用是保证实验的可重复性
        example_batch['observations'],
        example_batch['actions'],   #把一个样例 batch 的 observation 和 action 传给 agent 的 create 方法，agent 可以根据这些信息来构建自己的网络结构等
        config, #把config传给agent，agent会根据config来构建自己的网络结构等
    )
    action_dim = example_batch["actions"].shape[-1]
    initial_replay_size = 0 if FLAGS.balanced_sampling else train_dataset.size
    occupancy_detector = None
    occupancy_detector_config = None
    if FLAGS.occupancy_detector:
        detector_network_config = {
            "hidden_dim": FLAGS.occupancy_hidden_dim,
            "num_hidden_layers": FLAGS.occupancy_num_hidden_layers,
            "learning_rate": FLAGS.occupancy_lr,
            "clip_grad_norm": OCCUPANCY_CLIP_GRAD_NORM,
        }
        occupancy_detector_config = {
            **detector_network_config,
            "batch_size": FLAGS.occupancy_batch_size,
            "start_size": FLAGS.occupancy_start_size,
            "update_interval": FLAGS.occupancy_update_interval,
            "updates_per_interval": (
                FLAGS.occupancy_updates_per_interval
            ),
        }
        occupancy_example_transition = {
            field: np.expand_dims(
                np.asarray(example_batch[field]), axis=0
            )
            for field in TRANSITION_FIELDS
        }
        occupancy_seed = (
            int(FLAGS.seed) + OCCUPANCY_SEED_OFFSET
        ) % (2**32)
        occupancy_detector = TransitionOccupancyDetector.create(
            seed=occupancy_seed,
            example_offline_transition=occupancy_example_transition,
            config=detector_network_config,
        )

    params = agent.network.params
    # filter all target network
    params = {k: v for k, v in params.items() if "target" not in k}

    print(params.keys())
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("param count:", param_count)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")
    if occupancy_detector is not None:
        prefixes.append("occupancy_detector")
    if bridge_runtime_config.enabled:
        prefixes.append("dynamics_shift_bridge")
    csv_loggers = {prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes}

    load_stage = None
    load_step = None
    online_checkpoint = None
    if os.path.isdir(FLAGS.save_dir):
        print("trying to load from", FLAGS.save_dir)
        if os.path.exists(os.path.join(FLAGS.save_dir, 'token.tk')):
            print("found existing completed run. Exiting...")
            exit()

        load_stage, load_step = read_progress(FLAGS.save_dir)
        validate_bridge_checkpoint_resume(
            bridge_runtime_config.enabled, load_stage
        )
        if load_stage == "offline":
            try:
                agent = restore_agent(
                    agent,
                    restore_path=FLAGS.save_dir,
                    restore_epoch=load_step,
                )
                restore_csv_loggers(csv_loggers, FLAGS.save_dir)
            except Exception as exc:
                print(f"failed to load previous offline run: {exc}")
                load_stage = None
                load_step = None
        elif load_stage == "online":
            if FLAGS.ogbench_dataset_dir is not None:
                raise NotImplementedError(
                    "Online checkpoint recovery with a custom ogbench_dataset_dir "
                    "is not supported because custom dataset rotation recovery is "
                    "not implemented."
                )

    if load_stage is None:
        print("failed to load prev run")
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        flag_dict = get_flag_dict()
        with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

    logger = LoggingHelper(
        csv_loggers=csv_loggers,
        wandb_logger=wandb,
    )

    # Offline RL
    if load_stage == "online":
        start_step = FLAGS.offline_steps + 1
        print(f"skipping offline training restored at online step {load_step}")
    elif load_stage == "offline" and load_step is not None:
        start_step = load_step + 1
        print(f"restoring from offline step {start_step}")
    else:
        start_step = 1

    for i in tqdm.tqdm(range(start_step, FLAGS.offline_steps + 1)):
        log_step = i

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=dataset_env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        if config['agent_name'] == 'rebrac':
            agent, offline_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
        else:
            agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # eval
        if i == FLAGS.offline_steps or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            evaluate_kwargs = dict(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            if bridge_runtime_config.enabled:
                # Dataset.sample_sequence() uses NumPy's process-global RNG.
                # Keep gain-dependent evaluation from changing subsequent
                # offline QAM batches in paired Bridge experiments.
                eval_info, _, _ = call_preserving_global_numpy_rng(
                    evaluate, **evaluate_kwargs
                )
            else:
                eval_info, _, _ = evaluate(**evaluate_kwargs)
            logger.log(eval_info, "eval", step=log_step)
            
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)
            save_csv_loggers(csv_loggers, FLAGS.save_dir)
            with open(os.path.join(FLAGS.save_dir, 'progress.tk'), 'w') as f:
                f.write(f"offline,{i}")

    # transition from offline to online
    bridge_runtime = None
    if bridge_runtime_config.enabled:
        print(
            "pretraining Dynamics-Shift Bridge on primitive offline "
            "transitions",
            flush=True,
        )
        bridge_runtime = DynamicsShiftBridgeRuntime.create(
            config=bridge_runtime_config,
            dataset=bridge_primitive_transitions,
            expected_action_shape=tuple(env.action_space.shape),
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            seed=FLAGS.seed,
        )
        print(
            "Bridge offline held-out MSE: "
            f"normalized={float(bridge_runtime.offline_eval['normalized_mse'])}, "
            f"raw={float(bridge_runtime.offline_eval['raw_mse'])}; "
            "normalization samples="
            f"{bridge_runtime.normalization_sample_count}",
            flush=True,
        )

    print(train_dataset.keys())
    print(train_dataset["observations"].shape)

    if not FLAGS.balanced_sampling:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=train_dataset.size + FLAGS.online_steps
        )
    else:
        replay_buffer = ReplayBuffer.create(example_batch, size=FLAGS.online_steps)

    recent_dynamics_buffer = None
    if FLAGS.recent_dynamics_capacity > 0:
        # Build from the physical single-transition ReplayBuffer layout, not
        # from a sequence sampled for action-chunk training. In particular,
        # actions here have the environment's single-step action shape.
        recent_transition_template = create_recent_transition_template(
            replay_buffer
        )
        recent_dynamics_buffer = RecentDynamicsBuffer.create(
            recent_transition_template,
            capacity=FLAGS.recent_dynamics_capacity,
        )

    if load_stage == "online":
        (
            agent,
            occupancy_detector,
            online_rng,
            online_checkpoint,
        ) = load_online_checkpoint(
            FLAGS.save_dir,
            agent,
            replay_buffer=replay_buffer,
            recent_dynamics_buffer=recent_dynamics_buffer,
            expected_env_name=FLAGS.env_name,
            expected_horizon_length=FLAGS.horizon_length,
            expected_balanced_sampling=FLAGS.balanced_sampling,
            expected_initial_replay_size=initial_replay_size,
            expected_action_dim=action_dim,
            expected_offline_steps=FLAGS.offline_steps,
            expected_recent_dynamics_capacity=(
                FLAGS.recent_dynamics_capacity
            ),
            expected_online_step=load_step,
            occupancy_detector=occupancy_detector,
            expected_occupancy_detector_config=(
                occupancy_detector_config
            ),
        )
        restore_csv_loggers(csv_loggers, FLAGS.save_dir)
        online_loop_start = online_start_step(online_checkpoint)
        last_saved_online_step = online_checkpoint["online_step"]
        print(f"restoring online training from step {online_loop_start}")
    else:
        online_loop_start = 1
        last_saved_online_step = 0

    # Online RL
    update_info = {}
    action_queue = [] # for action chunking
    ob, _ = env.reset()

    for i in tqdm.tqdm(range(online_loop_start, FLAGS.online_steps + 1)):
        log_step = FLAGS.offline_steps + i
        online_rng, key = jax.random.split(online_rng)

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=dataset_env,
            )
            train_dataset = process_train_dataset(train_dataset)
            size = train_dataset.size
            
            if FLAGS.balanced_sampling:
                pass
            else:
                for k in train_dataset:
                    replay_buffer[k][:size] = train_dataset[k][:]

        # the action chunk is executed fully
        if len(action_queue) == 0:

            if FLAGS.balanced_sampling and i < FLAGS.start_training:
                action = np.random.rand(action_dim) * 2. - 1.
                action = np.clip(action, -1., 1.)
            else:
                action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)

        if bridge_runtime is None:
            next_ob, int_reward, terminated, truncated, info = env.step(
                action
            )
        else:
            (
                environment_result,
                executed_action,
                _corrected_action,
                _bridge_correction_metrics,
            ) = bridge_step_environment(
                bridge_runtime,
                env,
                ob,
                action,
            )
            (
                next_ob,
                int_reward,
                terminated,
                truncated,
                info,
            ) = environment_result
            action = executed_action
        done = terminated or truncated

        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"): # for cubes
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        if recent_dynamics_buffer is not None:
            # ReplayBuffer assignment already stores these values in its field
            # dtypes. Normalize explicitly so the strict recent buffer records
            # exactly that single-transition storage layout.
            transition = recent_dynamics_buffer.prepare_transition(transition)
        replay_buffer.add_transition(transition)
        if recent_dynamics_buffer is not None:
            recent_dynamics_buffer.add_transition(transition)
        if bridge_runtime is not None:
            bridge_runtime.maybe_update_online(
                online_step=i,
                recent_buffer=recent_dynamics_buffer,
            )

        if (
            occupancy_detector is not None
            and should_update_occupancy_detector(
                online_step=i,
                recent_size=recent_dynamics_buffer.size,
                enabled=True,
                start_size=FLAGS.occupancy_start_size,
                update_interval=FLAGS.occupancy_update_interval,
            )
        ):
            occupancy_train_metrics = []
            for _ in range(FLAGS.occupancy_updates_per_interval):
                (
                    occupancy_detector,
                    offline_transitions,
                    online_transitions,
                ) = sample_occupancy_transition_batches(
                    occupancy_detector,
                    train_dataset,
                    recent_dynamics_buffer,
                    FLAGS.occupancy_batch_size,
                )
                occupancy_detector, detector_metrics = (
                    occupancy_detector.update(
                        offline_transitions,
                        online_transitions,
                    )
                )
                occupancy_train_metrics.append(detector_metrics)
            averaged_train_metrics = average_occupancy_metrics(
                occupancy_train_metrics
            )

            # Freshly resample after the full burst for evaluation. Sampling
            # with replacement does not guarantee that evaluation samples are
            # disjoint from training samples.
            (
                occupancy_detector,
                eval_offline_transitions,
                eval_online_transitions,
            ) = sample_occupancy_transition_batches(
                occupancy_detector,
                train_dataset,
                recent_dynamics_buffer,
                FLAGS.occupancy_batch_size,
            )
            eval_metrics = occupancy_detector.evaluate(
                eval_offline_transitions,
                eval_online_transitions,
            )
            logged_occupancy_metrics = {
                **{
                    f"train/{name}": value
                    for name, value in averaged_train_metrics.items()
                },
                **{
                    f"eval/{name}": value
                    for name, value in eval_metrics.items()
                },
            }
            logger.log(
                logged_occupancy_metrics,
                "occupancy_detector",
                step=log_step,
            )
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:

            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample_sequence(config['batch_size'] // 2 * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
                replay_batch = replay_buffer.sample_sequence(FLAGS.utd_ratio * config['batch_size'] // 2, 
                    sequence_length=FLAGS.horizon_length, discount=discount)
                
                batch = {k: np.concatenate([
                    dataset_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + dataset_batch[k].shape[1:]), 
                    replay_batch[k].reshape((FLAGS.utd_ratio, config["batch_size"] // 2) + replay_batch[k].shape[1:])], axis=1) for k in dataset_batch}
                
            else:
                batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                            sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            if config['agent_name'] == 'rebrac':
                agent, update_info["online_agent"] = agent.batch_update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}
        if (
            bridge_runtime is not None
            and (
                i % FLAGS.log_interval == 0
                or i == FLAGS.online_steps
            )
        ):
            logger.log(
                bridge_runtime.log_row(
                    online_step=i,
                    recent_buffer_size=recent_dynamics_buffer.size,
                ),
                "dynamics_shift_bridge",
                step=log_step,
            )

        if i == FLAGS.online_steps or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            evaluate_kwargs = dict(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            if bridge_runtime_config.enabled:
                eval_info, _, _ = call_preserving_global_numpy_rng(
                    evaluate, **evaluate_kwargs
                )
            else:
                eval_info, _, _ = evaluate(**evaluate_kwargs)
            logger.log(eval_info, "eval", step=log_step)

        if should_save_online_checkpoint(
            online_step=i,
            online_save_interval=FLAGS.online_save_interval,
            last_saved_online_step=last_saved_online_step,
            done=done,
            action_queue=action_queue,
        ):
            save_csv_loggers(csv_loggers, FLAGS.save_dir)
            save_online_checkpoint(
                save_dir=FLAGS.save_dir,
                agent=agent,
                replay_buffer=replay_buffer,
                online_rng=online_rng,
                online_step=i,
                offline_steps=FLAGS.offline_steps,
                balanced_sampling=FLAGS.balanced_sampling,
                initial_replay_size=initial_replay_size,
                action_dim=action_dim,
                horizon_length=FLAGS.horizon_length,
                env_name=FLAGS.env_name,
                done=done,
                action_queue=action_queue,
                recent_dynamics_buffer=recent_dynamics_buffer,
                recent_dynamics_capacity=FLAGS.recent_dynamics_capacity,
                occupancy_detector=occupancy_detector,
                occupancy_detector_config=occupancy_detector_config,
            )
            last_saved_online_step = i
            print(f"saved online checkpoint at episode boundary step {i}")

    # a token to indicate a successfully finished run
    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url or "")

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    wandb.finish()

    # cleanup
    if FLAGS.auto_cleanup:
        all_files = os.listdir(FLAGS.save_dir)
        for relative_path in all_files:
            full_path = os.path.join(FLAGS.save_dir, relative_path)
            if os.path.isfile(full_path) and relative_path.startswith("params"):
                print(f"removing {full_path}")
                os.remove(full_path)
        online_checkpoint_path = os.path.join(FLAGS.save_dir, CHECKPOINT_FILENAME)
        if os.path.isfile(online_checkpoint_path):
            print(f"removing {online_checkpoint_path}")
            os.remove(online_checkpoint_path)

if __name__ == '__main__':
    app.run(main)
