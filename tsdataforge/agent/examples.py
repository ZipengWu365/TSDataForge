from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from textwrap import dedent
from typing import Iterable


@dataclass(frozen=True)
class ExampleRecipe:
    example_id: str
    title: str
    summary: str
    goal: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    audience: tuple[str, ...] = field(default_factory=tuple)
    code: str = ""
    learnings: tuple[str, ...] = field(default_factory=tuple)
    filename: str | None = None
    category: str = "general"
    difficulty: str = "intro"
    outputs: tuple[str, ...] = field(default_factory=tuple)
    related_api: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


_CATALOG: tuple[ExampleRecipe, ...] = (
    ExampleRecipe(
        example_id="quickstart_univariate",
        title="5 分钟快速上手：trend + seasonal + noise",
        summary="最短路径理解 generate_series、trace 和结构组合。",
        goal="快速看懂 TSDataForge 的单条序列工作流。",
        keywords=("quickstart", "generate_series", "trend", "seasonal", "noise", "intro"),
        audience=("new_user", "researcher", "agent"),
        filename="quickstart_univariate.py",
        category="quickstart",
        difficulty="intro",
        outputs=("GeneratedSeries", "trace"),
        related_api=("generate_series", "LinearTrend", "MultiSineSeasonality", "WhiteGaussianNoise"),
        code=dedent(
            '''
            from tsdataforge import generate_series
            from tsdataforge.primitives import LinearTrend, MultiSineSeasonality, WhiteGaussianNoise

            sample = generate_series(
                length=512,
                components=[
                    LinearTrend(slope=0.01, intercept=0.2),
                    MultiSineSeasonality(freqs=(24.0, 96.0), amps=(1.0, 0.25)),
                    WhiteGaussianNoise(std=0.1),
                ],
                seed=42,
            )
            print(sample.values.shape)
            print(sample.trace.tags)
            '''
        ).strip(),
        learnings=(
            "最小 API 足够短，适合 agent 直接生成。",
            "trace 会保留结构标签与中间信息，后续 taskify / EDA 可直接复用。",
        ),
    ),
    ExampleRecipe(
        example_id="quickstart_dataset_pipeline",
        title="从基础数据集到保存资产：一条完整最短链路",
        summary="generate_series_dataset -> taskify -> save，适合 onboarding。",
        goal="给新用户一条最不容易出错的端到端路径。",
        keywords=("quickstart", "dataset", "taskify", "save", "onboarding"),
        audience=("new_user", "maintainer", "agent"),
        filename="quickstart_dataset_pipeline.py",
        category="quickstart",
        difficulty="intro",
        outputs=("SeriesDataset", "TaskDataset", "README", "card"),
        related_api=("generate_series_dataset", "SeriesDataset", "TaskDataset"),
        code=dedent(
            '''
            from pathlib import Path
            from tsdataforge import generate_series_dataset

            base = generate_series_dataset(
                structures=["trend_seasonal_noise", "regime_switch"],
                n_series=24,
                length=192,
                seed=0,
            )
            forecast = base.taskify(task="forecasting", horizon=24)
            forecast.save(Path("saved_forecast_dataset"))
            print(forecast.X.shape, forecast.y.shape)
            '''
        ).strip(),
        learnings=(
            "先生成基础数据集，再派生任务，最有利于复用。",
            "save() 会把数据、README、context、card 一起落盘。",
        ),
    ),
    ExampleRecipe(
        example_id="dataset_handoff_bundle",
        title="一键生成 dataset handoff bundle",
        summary="把 report、context、card、manifest 打成一个可分享目录。",
        goal="给人和 agent 一条最短的 dataset -> report -> handoff 路径。",
        keywords=("handoff", "report", "bundle", "eda", "agent", "quickstart"),
        audience=("new_user", "maintainer", "agent"),
        filename="dataset_handoff_bundle.py",
        category="agent",
        difficulty="intro",
        outputs=("report.html", "dataset_context", "dataset_card", "handoff_bundle"),
        related_api=("generate_series_dataset", "build_dataset_handoff_bundle"),
        code=dedent(
            '''
            from tsdataforge import generate_series_dataset, build_dataset_handoff_bundle

            base = generate_series_dataset(
                structures=["trend_seasonal_noise", "regime_switch"],
                n_series=24,
                length=192,
                seed=0,
            )
            bundle = build_dataset_handoff_bundle(
                base,
                output_dir="dataset_handoff_bundle",
                include_report=True,
            )
            print(bundle.output_dir)
            '''
        ).strip(),
        learnings=(
            "这是最短的 outcome-first happy path。",
            "适合 GitHub 演示、内部交接和 agent workflow。",
        ),
    ),
    ExampleRecipe(
        example_id="taskify_forecasting",
        title="把基础数据集任务化成 forecasting",
        summary="SeriesDataset -> TaskDataset 的核心路径。",
        goal="把基础数据集变成可直接训练的预测数据集。",
        keywords=("forecasting", "taskify", "dataset", "window", "horizon"),
        audience=("researcher", "ml_engineer", "agent"),
        filename="taskify_forecasting.py",
        category="taskification",
        difficulty="intro",
        outputs=("TaskDataset",),
        related_api=("generate_series_dataset", "taskify_dataset", "TaskDataset"),
        code=dedent(
            '''
            from tsdataforge import generate_series_dataset

            base = generate_series_dataset(
                structures=["trend_seasonal_noise", "white_noise"],
                n_series=32,
                length=256,
                seed=0,
            )
            forecasting = base.taskify(task="forecasting", horizon=32)
            print(forecasting.X.shape, forecasting.y.shape)
            print(forecasting.schema)
            '''
        ).strip(),
        learnings=(
            "taskify 把 raw dataset 和任务语义解耦。",
            "schema 能直接告诉 agent X/y 的含义，避免重新解释。",
        ),
    ),
    ExampleRecipe(
        example_id="classification_benchmark",
        title="结构分类 benchmark：taxonomy 直接变标签",
        summary="把不同 structure recipe 直接生成为分类任务。",
        goal="快速构建结构识别 benchmark。",
        keywords=("classification", "taxonomy", "benchmark", "label"),
        audience=("researcher", "benchmark", "agent"),
        filename="classification_benchmark.py",
        category="taskification",
        difficulty="intro",
        outputs=("TaskDataset", "label_names"),
        related_api=("generate_dataset",),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="classification",
                structures=["white_noise", "trend_seasonal_noise", "regime_switch"],
                n_series=60,
                length=192,
                seed=1,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.label_names)
            '''
        ).strip(),
        learnings=(
            "taxonomy 本身就能成为可解释的标签空间。",
            "balanced 采样会让类别更均匀。",
        ),
    ),
    ExampleRecipe(
        example_id="masked_reconstruction",
        title="自监督 masked reconstruction",
        summary="构造被 mask 的输入和对应重建目标。",
        goal="给预训练或表示学习准备 mask 恢复任务。",
        keywords=("masked", "self-supervised", "pretraining", "reconstruction"),
        audience=("ssl", "researcher", "agent"),
        filename="masked_reconstruction.py",
        category="taskification",
        difficulty="intro",
        outputs=("TaskDataset", "masks"),
        related_api=("generate_dataset",),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="masked_reconstruction",
                structures=["trend_seasonal_noise"],
                n_series=16,
                length=128,
                mask_ratio=0.2,
                seed=4,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.masks.keys())
            '''
        ).strip(),
        learnings=(
            "同一基础结构可以同时服务预测和自监督任务。",
            "masks 会显式进入 TaskDataset，而不是藏在 notebook 里。",
        ),
    ),
    ExampleRecipe(
        example_id="anomaly_detection",
        title="异常检测任务：自动注入异常并给标签",
        summary="构造点异常/段异常的检测 benchmark。",
        goal="快速准备 anomaly detection 数据集。",
        keywords=("anomaly", "detection", "label", "benchmark"),
        audience=("researcher", "ops", "agent"),
        filename="anomaly_detection.py",
        category="taskification",
        difficulty="intro",
        outputs=("TaskDataset",),
        related_api=("generate_dataset",),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="anomaly_detection",
                structures=["trend_seasonal_noise"],
                n_series=24,
                length=192,
                anomaly_rate=0.03,
                seed=6,
            )
            print(ds.X.shape, ds.y.shape)
            '''
        ).strip(),
        learnings=(
            "异常任务不需要另写标签器。",
            "anomaly_rate 是最重要的一个实验控制旋钮。",
        ),
    ),
    ExampleRecipe(
        example_id="change_point_detection",
        title="变化点检测：regime switch 直接给真值",
        summary="利用 trace 中的 regime / changepoint 信息构造 CPD benchmark。",
        goal="做 change-point detection 实验。",
        keywords=("change point", "regime", "cpd", "trace"),
        audience=("researcher", "agent"),
        filename="change_point_detection.py",
        category="taskification",
        difficulty="intro",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "RegimeSwitch"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="change_point_detection",
                structures=["regime_switch"],
                n_series=24,
                length=192,
                seed=7,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.y.sum(axis=1)[:5])
            '''
        ).strip(),
        learnings=(
            "trace 是 task 化的关键资产。",
            "regime 类结构非常适合拿来做变化点任务。",
        ),
    ),
    ExampleRecipe(
        example_id="contrastive_pairs",
        title="contrastive pairs：同结构不同噪声，对比学习直接开跑",
        summary="构造正负样本对做表示学习。",
        goal="用统一接口做对比学习实验。",
        keywords=("contrastive", "ssl", "representation", "pairs"),
        audience=("ssl", "researcher", "agent"),
        filename="contrastive_pairs.py",
        category="taskification",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset",),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="contrastive",
                structures=["trend_seasonal_noise", "regime_switch"],
                n_series=20,
                length=160,
                seed=10,
            )
            print(ds.X.shape)
            print(ds.schema)
            '''
        ).strip(),
        learnings=(
            "结构空间天然适合构造 hard negatives。",
            "contrastive 的关键不是样本量，而是结构差异定义得是否清楚。",
        ),
    ),
    ExampleRecipe(
        example_id="real_series_eda",
        title="真实序列描述 + EDA 报告",
        summary="从真实数据抽取 tags、解释和 HTML 报告。",
        goal="快速判断真实序列的结构类型和建模方向。",
        keywords=("eda", "describe", "real data", "report", "explain"),
        audience=("analyst", "researcher", "agent"),
        filename="real_series_eda.py",
        category="eda",
        difficulty="intro",
        outputs=("SeriesDescription", "EDAReport"),
        related_api=("describe_series", "generate_eda_report"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import describe_series, generate_eda_report

            t = np.arange(512, dtype=float)
            y = np.sin(2 * np.pi * t / 24.0) + 0.15 * np.random.default_rng(0).normal(size=len(t))

            desc = describe_series(y, t)
            print(desc.inferred_tags)
            rep = generate_eda_report(y, t, output_path="series_report.html")
            print(rep.output_path)
            '''
        ).strip(),
        learnings=(
            "先描述再建模，可以减少瞎试。",
            "EDA 报告适合团队沟通，也适合给 agent 先读。",
        ),
    ),
    ExampleRecipe(
        example_id="dataset_eda",
        title="整个时间序列数据集的 EDA 报告",
        summary="批量数据集的 tag 频次、signature 覆盖、长度分布一页看完。",
        goal="快速盘点一个数据集到底长什么样。",
        keywords=("dataset", "eda", "coverage", "signature", "report"),
        audience=("maintainer", "researcher", "agent"),
        filename="dataset_eda.py",
        category="eda",
        difficulty="intro",
        outputs=("EDAReport",),
        related_api=("generate_dataset_eda_report", "describe_dataset"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import generate_dataset_eda_report

            X = np.random.default_rng(0).normal(size=(48, 128))
            rep = generate_dataset_eda_report(X, output_path="dataset_report.html")
            print(rep.output_path)
            '''
        ).strip(),
        learnings=(
            "数据集级覆盖率统计对 benchmark 设计非常重要。",
            "signature 分桶能帮你快速发现数据集偏置。",
        ),
    ),
    ExampleRecipe(
        example_id="spec_from_real_series",
        title="从真实序列推一个 synthetic spec seed",
        summary="real -> describe -> suggest_spec -> generate_series 的闭环。",
        goal="基于真实数据快速做 matching synthetic baseline。",
        keywords=("real data", "suggest_spec", "matching", "synthetic"),
        audience=("researcher", "analyst", "agent"),
        filename="spec_from_real_series.py",
        category="eda",
        difficulty="intermediate",
        outputs=("SeriesSpec",),
        related_api=("describe_series", "suggest_spec", "generate_series"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import describe_series, suggest_spec, generate_series

            t = np.arange(384, dtype=float)
            y = 0.01 * t + np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.default_rng(2).normal(size=len(t))
            desc = describe_series(y, t)
            spec = suggest_spec(desc)
            sample = generate_series(length=384, spec=spec, seed=123)
            print(desc.inferred_tags)
            print(sample.values.shape)
            '''
        ).strip(),
        learnings=(
            "real -> spec 是连接真实世界和 synthetic research 的关键闭环。",
            "spec seed 不必完美，只要能稳定表达主要结构。",
        ),
    ),
    ExampleRecipe(
        example_id="robust_observation_variants",
        title="同一 latent，不同观测机制：鲁棒性实验一键搭起来",
        summary="不规则采样、块状缺失、测量噪声和裁剪统一进入 ObservationSpec。",
        goal="评估模型对观测机制变化的鲁棒性。",
        keywords=("observation", "robustness", "irregular", "missing", "noise"),
        audience=("researcher", "benchmark", "agent"),
        filename="robust_observation_variants.py",
        category="eda",
        difficulty="intermediate",
        outputs=("GeneratedSeries",),
        related_api=("ObservationSpec", "IrregularSampling", "BlockMissing", "MeasurementNoise", "Clamp"),
        code=dedent(
            '''
            from tsdataforge import generate_series, ObservationSpec
            from tsdataforge.observation import IrregularSampling, BlockMissing, MeasurementNoise, Clamp
            from tsdataforge.primitives import LinearTrend, MultiSineSeasonality

            obs = ObservationSpec(
                sampling=IrregularSampling(dt=1.0, jitter=0.2),
                missing=BlockMissing(rate=0.08, block_min=3, block_max=12),
                measurement_noise=MeasurementNoise(std=0.05),
                transforms=(Clamp(-3.0, 3.0),),
            )
            sample = generate_series(
                length=256,
                components=[LinearTrend(0.01), MultiSineSeasonality(freqs=(24.0,), amps=(1.0,))],
                observation=obs,
                seed=8,
            )
            print(sample.values.shape)
            print(sample.trace.masks.keys())
            '''
        ).strip(),
        learnings=(
            "观测机制和 latent structure 应该解耦。",
            "很多真实世界差异来自观测层，而不是潜在动力学。",
        ),
    ),
    ExampleRecipe(
        example_id="external_rollout_wrap",
        title="接入外部仿真/真实 rollout，不重复造仿真轮子",
        summary="把 MuJoCo / PyBullet / ROS 导出的序列包装成 TSDataForge 对象。",
        goal="统一外部控制时序与 TSDataForge 的 EDA / taskify 工作流。",
        keywords=("external", "rollout", "robotics", "integration", "wrap"),
        audience=("robotics", "control", "agent"),
        filename="external_rollout_wrap.py",
        category="integration",
        difficulty="intro",
        outputs=("GeneratedSeries",),
        related_api=("wrap_external_series", "generate_eda_report"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge.integrations import wrap_external_series

            t = np.linspace(0, 5, 1000)
            values = np.stack([np.sin(2 * np.pi * 1.2 * t), np.cos(2 * np.pi * 1.2 * t)], axis=1)

            gs = wrap_external_series(
                values,
                t,
                channel_names=["q", "dq"],
                tags=("external", "control", "rollout"),
                meta={"source": "mujoco"},
            )
            print(gs.values.shape)
            print(gs.trace.tags)
            '''
        ).strip(),
        learnings=(
            "TSDataForge 负责 taxonomy / EDA / taskify，仿真本体留给成熟外部栈。",
            "这样库的职责更清晰，也更容易长期维护。",
        ),
    ),
    ExampleRecipe(
        example_id="system_identification",
        title="控制输入-输出对的 system identification",
        summary="直接拿到 past [u,y] -> future y 的数据格式。",
        goal="做状态空间或控制对象辨识。",
        keywords=("system identification", "state space", "control", "io"),
        audience=("control", "ml_engineer", "agent"),
        filename="system_identification.py",
        category="control_causal",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "LinearStateSpace"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="system_identification",
                structures=["linear_state_space_io"],
                n_series=12,
                length=160,
                horizon=24,
                seed=11,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.aux["u"].shape, ds.aux["x"].shape)
            '''
        ).strip(),
        learnings=(
            "system identification 不是单独的一套生成逻辑，而是 taskify 视角。",
            "同一基础序列还可以再变成 forecasting 或 counterfactual 任务。",
        ),
    ),
    ExampleRecipe(
        example_id="event_control_detection",
        title="带控制的 event detection 数据集",
        summary="event-triggered control 或 bursty 序列直接转成事件检测任务。",
        goal="快速生成事件检测 benchmark。",
        keywords=("event detection", "control", "trigger", "benchmark"),
        audience=("control", "researcher", "agent"),
        filename="event_control_detection.py",
        category="control_causal",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "EventTriggeredController"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="event_detection",
                structures=["event_triggered_control", "bursty_event_response"],
                n_series=16,
                length=192,
                seed=9,
            )
            print(ds.X.shape, ds.y.shape)
            '''
        ).strip(),
        learnings=(
            "trace 自动提供事件真值，省掉手写标签。",
            "控制类时序和普通 bursty 事件可以在同一任务协议下比较。",
        ),
    ),
    ExampleRecipe(
        example_id="causal_response",
        title="causal response：历史 treatment/协变量 -> future outcome",
        summary="适合做时间依赖的干预响应预测。",
        goal="用统一接口生成因果响应任务。",
        keywords=("causal", "treatment", "outcome", "response", "counterfactual"),
        audience=("causal", "researcher", "agent"),
        filename="causal_response.py",
        category="control_causal",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "CausalTreatmentOutcome"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="causal_response",
                structures=["causal_treatment_outcome"],
                n_series=24,
                length=256,
                horizon=32,
                seed=2,
            )
            print(ds.X.shape, ds.y.shape)
            '''
        ).strip(),
        learnings=(
            "因果任务和预测任务可以共享同一 TaskDataset 协议。",
            "只要 trace 里有因果真值，就还能派生 ITE / discovery。",
        ),
    ),
    ExampleRecipe(
        example_id="policy_counterfactual",
        title="policy / intervention / counterfactual 成对 rollout",
        summary="对同一系统生成 factual vs counterfactual 序列。",
        goal="做反事实评估、policy 比较、干预检测。",
        keywords=("policy", "counterfactual", "intervention", "control", "causal"),
        audience=("control", "causal", "agent"),
        filename="policy_counterfactual.py",
        category="control_causal",
        difficulty="advanced",
        outputs=("CounterfactualPair",),
        related_api=("generate_counterfactual_pair", "with_policy", "with_intervention"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import SeriesSpec, generate_counterfactual_pair
            from tsdataforge.dynamics import PolicyControlledStateSpace
            from tsdataforge.interventions import InterventionSpec
            from tsdataforge.policies import ConstantPolicy, LinearFeedbackPolicy

            A = 0.9 * np.eye(2)
            B = np.ones((2, 1)) * 0.2
            C = np.eye(2)
            policy = LinearFeedbackPolicy(K=np.array([[0.6, -0.2]]), source="state", name="feedback")
            alt = ConstantPolicy(action_value=(0.0,), name="zero")

            spec = SeriesSpec(
                latent=PolicyControlledStateSpace(A=A, B=B, C=C, policy=policy, counterfactual_policies=(alt,)),
                structure_id="policy_controlled_state_space",
            )
            pair = generate_counterfactual_pair(
                spec=spec,
                length=120,
                seed=7,
                intervention=InterventionSpec(target="input", index=0, start=0.25, end=0.5, value=0.8),
            )
            print(pair.factual.values.shape, pair.counterfactual.values.shape)
            '''
        ).strip(),
        learnings=(
            "counterfactual pair 是 agent 做比较式分析时最省 token 的输入单位。",
            "你只需要读 factual/counterfactual 差异，不必把所有中间 trace 都塞进上下文。",
        ),
    ),
    ExampleRecipe(
        example_id="policy_value_estimation",
        title="policy value estimation：从 rollout 到 return",
        summary="直接生成带 reward 的任务数据集。",
        goal="做离线策略评估或 return 预测。",
        keywords=("policy", "value", "reward", "offline rl", "control"),
        audience=("control", "rl", "agent"),
        filename="policy_value_estimation.py",
        category="control_causal",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "PolicyControlledStateSpace"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="policy_value_estimation",
                structures=["policy_controlled_state_space"],
                n_series=16,
                length=128,
                seed=12,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.y[:5])
            '''
        ).strip(),
        learnings=(
            "policy 任务不止是 rollout，还可以直接 taskify 成 return 目标。",
            "reward / return 应该进入 schema，而不是隐藏在自定义代码里。",
        ),
    ),
    ExampleRecipe(
        example_id="intervention_detection",
        title="intervention detection：检测何时发生了外部干预",
        summary="利用 trace 中的 intervention mask 构造检测任务。",
        goal="识别控制/因果系统中的干预发生时刻。",
        keywords=("intervention", "detection", "causal", "control"),
        audience=("causal", "control", "agent"),
        filename="intervention_detection.py",
        category="control_causal",
        difficulty="intermediate",
        outputs=("TaskDataset",),
        related_api=("generate_dataset", "InterventionSpec"),
        code=dedent(
            '''
            from tsdataforge import generate_dataset

            ds = generate_dataset(
                task="intervention_detection",
                structures=["policy_controlled_state_space"],
                n_series=12,
                length=160,
                seed=14,
            )
            print(ds.X.shape, ds.y.shape)
            print(ds.y.sum(axis=1)[:3])
            '''
        ).strip(),
        learnings=(
            "intervention 与 event 都是序列级别的重要监督信号。",
            "这个任务对 agent 驱动的 debug 也很有价值。",
        ),
    ),
    ExampleRecipe(
        example_id="real_dataset_taskify",
        title="真实数据集直接 taskify 成预测/分类/因果格式",
        summary="把外部数组快速纳入统一任务协议。",
        goal="统一处理真实数据和合成数据。",
        keywords=("real data", "taskify", "from arrays", "forecasting", "causal"),
        audience=("analyst", "ml_engineer", "agent"),
        filename="real_dataset_taskify.py",
        category="integration",
        difficulty="intro",
        outputs=("SeriesDataset", "TaskDataset"),
        related_api=("SeriesDataset", "taskify_dataset"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge.datasets import SeriesDataset

            real_values = np.random.default_rng(0).normal(size=(32, 160, 4))
            real_ds = SeriesDataset.from_arrays(real_values, dataset_id="my_real_dataset")
            forecast = real_ds.taskify(task="forecasting", window=96, horizon=24, stride=24)
            print(forecast.X.shape, forecast.y.shape)
            '''
        ).strip(),
        learnings=(
            "TSDataForge 不要求所有数据都先被生成；真实数据也能直接进入统一协议。",
            "这对 agent 编排特别重要，因为流程更稳定。",
        ),
    ),
    ExampleRecipe(
        example_id="agent_context_pack",
        title="给 agent 的紧凑上下文包",
        summary="把 series/dataset/task 压成小而稳的结构化上下文。",
        goal="减少 token 浪费，让 agent 更容易拿到关键信息。",
        keywords=("agent", "context", "token", "compact", "schema"),
        audience=("agent", "platform", "maintainer"),
        filename="agent_context_pack.py",
        category="agent",
        difficulty="intro",
        outputs=("AgentContextPack",),
        related_api=("build_dataset_context", "build_task_context", "build_agent_context"),
        code=dedent(
            '''
            from tsdataforge import generate_series_dataset
            from tsdataforge.agent import build_dataset_context

            base = generate_series_dataset(structures=["trend_seasonal_noise", "causal_varx"], n_series=20, length=128, seed=0)
            pack = build_dataset_context(base, budget="small", goal="prepare forecasting and causal experiments")
            print(pack.to_dict())
            print(pack.to_markdown())
            '''
        ).strip(),
        learnings=(
            "给 agent 的不是原始数组，而是 compact schema + tags + next_actions。",
            "这比把完整 README 丢进上下文更稳、更省 token。",
        ),
    ),
    ExampleRecipe(
        example_id="dataset_cards",
        title="自动生成 dataset/task card",
        summary="保存数据时同时保存 README 和 JSON 卡片，方便 agent 和人类同时消费。",
        goal="让数据资产自己带说明书。",
        keywords=("card", "manifest", "save", "agent", "readme"),
        audience=("platform", "maintainer", "agent"),
        filename="dataset_cards.py",
        category="agent",
        difficulty="intro",
        outputs=("ArtifactCard", "README"),
        related_api=("build_series_dataset_card", "build_task_dataset_card"),
        code=dedent(
            '''
            from pathlib import Path
            from tsdataforge import generate_dataset
            from tsdataforge.agent import build_task_dataset_card

            ds = generate_dataset(task="forecasting", structures=["trend_seasonal_noise"], n_series=8, length=96, horizon=16, seed=3)
            card = build_task_dataset_card(ds)
            print(card.to_markdown())
            ds.save(Path("saved_forecasting_dataset"))
            '''
        ).strip(),
        learnings=(
            "manifest + README 能显著降低后续团队沟通成本。",
            "agent 只要读 card.json 就能知道怎么用这个数据集。",
        ),
    ),
    ExampleRecipe(
        example_id="api_reference_overview",
        title="一键导出 API Reference 与 JSON manifest",
        summary="把公共 API 表层整理成结构化清单。",
        goal="补齐 API 文档，并给 agent 一个可消费的 API manifest。",
        keywords=("api", "reference", "manifest", "docs", "agent"),
        audience=("maintainer", "platform", "agent"),
        filename="api_reference_overview.py",
        category="agent",
        difficulty="intro",
        outputs=("APIReference", "json", "markdown"),
        related_api=("build_api_reference", "save_api_reference", "render_api_reference_markdown"),
        code=dedent(
            '''
            from pathlib import Path
            from tsdataforge.agent import build_api_reference, save_api_reference

            ref = build_api_reference()
            save_api_reference(ref, Path("api_reference.json"))
            save_api_reference(ref, Path("api_reference.md"))
            print(ref.n_symbols)
            print(ref.categories[0].title)
            '''
        ).strip(),
        learnings=(
            "API 文档也应该是程序可消费的，而不只是人类阅读。",
            "release 时把 API manifest 一起发出去，对 agent 很友好。",
        ),
    ),
    ExampleRecipe(
        example_id="docs_site_generation",
        title="一键生成 docs / cookbook / agent playbook 静态站点",
        summary="把库的说明、案例、FAQ、API 直接生成为可分发的 HTML。",
        goal="做扩散、教学和社区 onboarding。",
        keywords=("docs", "cookbook", "site", "onboarding", "launch", "faq", "api"),
        audience=("maintainer", "community", "agent"),
        filename="docs_site_generation.py",
        category="launch",
        difficulty="intro",
        outputs=("DocsSiteResult", "html"),
        related_api=("generate_docs_site",),
        code=dedent(
            '''
            from tsdataforge.agent import generate_docs_site

            site = generate_docs_site("tsdataforge_docs_site")
            print(site.pages)
            print(site.example_pages[:3])
            print(site.api_pages[:2])
            '''
        ).strip(),
        learnings=(
            "功能合理之后，文档和案例决定扩散速度。",
            "静态站点很轻，适合快速托管和分享。",
        ),
    ),
    ExampleRecipe(
        example_id="example_routing",
        title="自然语言目标 -> 推荐案例 -> 最短改写",
        summary="用例路由器是非常适合 agent 的第一步。",
        goal="让新用户和 agent 都能快速找到最接近的示例。",
        keywords=("recommend", "examples", "routing", "agent", "goal"),
        audience=("agent", "maintainer", "new_user"),
        filename="example_routing.py",
        category="launch",
        difficulty="intro",
        outputs=("ExampleRecipe",),
        related_api=("recommend_examples", "example_catalog"),
        code=dedent(
            '''
            from tsdataforge.agent import recommend_examples

            recs = recommend_examples("I need causal response forecasting with real data", top_k=3)
            for ex in recs:
                print(ex.example_id, ex.title)
            '''
        ).strip(),
        learnings=(
            "案例路由通常比长篇解释更有效。",
            "自然语言入口对扩散尤其重要。",
        ),
    ),
)


_EXAMPLE_EN: dict[str, dict[str, object]] = {
    "quickstart_univariate": {
        "title": "5-minute quickstart: trend + seasonal + noise",
        "summary": "The shortest path to understand generate_series, trace, and structure composition.",
        "goal": "Learn the single-series TSDataForge workflow in one small example.",
        "learnings": (
            "The minimal API is short enough for direct agent generation.",
            "Trace keeps structure tags and intermediate information for later taskify / EDA reuse.",
        ),
    },
    "quickstart_dataset_pipeline": {
        "title": "From base dataset to saved assets: the shortest end-to-end path",
        "summary": "generate_series_dataset -> taskify -> save, ideal for onboarding.",
        "goal": "Give new users one end-to-end path that is hard to misuse.",
        "learnings": (
            "Generate a base dataset first, then derive tasks for maximum reuse.",
            "save() writes data, README, context, and card together.",
        ),
    },
    "taskify_forecasting": {
        "title": "Turn a base dataset into a forecasting task",
        "summary": "The core SeriesDataset -> TaskDataset path.",
        "goal": "Convert a base dataset into a directly trainable forecasting dataset.",
        "learnings": (
            "taskify decouples raw datasets from task semantics.",
            "schema tells agents exactly what X/y mean.",
        ),
    },
    "classification_benchmark": {
        "title": "Structure classification benchmark: taxonomy directly becomes labels",
        "summary": "Generate a classification task directly from structure recipes.",
        "goal": "Build an interpretable structure-recognition benchmark quickly.",
        "learnings": (
            "The taxonomy itself can serve as an explainable label space.",
            "Balanced sampling is usually the right default for classification benchmarks.",
        ),
    },
    "masked_reconstruction": {
        "title": "Self-supervised masked reconstruction",
        "summary": "Create masked inputs and matching reconstruction targets.",
        "goal": "Prepare a masked recovery task for pretraining or representation learning.",
        "learnings": (
            "The same base structures can support forecasting and self-supervised tasks.",
            "Masks live explicitly in TaskDataset instead of being hidden in a notebook.",
        ),
    },
    "anomaly_detection": {
        "title": "Anomaly detection: inject anomalies and get labels automatically",
        "summary": "Create point-anomaly and segment-anomaly benchmarks.",
        "goal": "Prepare anomaly-detection datasets quickly.",
        "learnings": (
            "You do not need a separate labeler for synthetic anomaly tasks.",
            "anomaly_rate is one of the most important experiment controls.",
        ),
    },
    "change_point_detection": {
        "title": "Change-point detection: regime switches come with ground truth",
        "summary": "Use regime / changepoint traces to build a CPD benchmark.",
        "goal": "Run change-point detection experiments with explicit targets.",
        "learnings": (
            "Trace is the key asset behind taskification.",
            "Regime-style structures are ideal for change-point tasks.",
        ),
    },
    "contrastive_pairs": {
        "title": "Contrastive pairs: same structure, different noise",
        "summary": "Build positive and negative pairs for representation learning.",
        "goal": "Run contrastive experiments through one consistent interface.",
        "learnings": (
            "Structure space is naturally suited to hard negatives.",
            "The key to contrastive quality is a clear definition of structural difference.",
        ),
    },
    "real_series_eda": {
        "title": "Describe a real series and generate an EDA report",
        "summary": "Extract tags, explanations, and an HTML report from real data.",
        "goal": "Decide quickly what a real series looks like and where modeling should start.",
        "learnings": (
            "Describe first, model second: it cuts down blind trial-and-error.",
            "EDA reports work for team communication and for agent grounding.",
        ),
    },
    "dataset_eda": {
        "title": "Dataset-level EDA for time-series collections",
        "summary": "See tag frequency, signature coverage, and length distribution in one page.",
        "goal": "Inventory what a dataset really looks like.",
        "learnings": (
            "Dataset-level coverage statistics are critical for benchmark design.",
            "Signature buckets help reveal dataset bias quickly.",
        ),
    },
    "spec_from_real_series": {
        "title": "Infer a synthetic spec seed from a real series",
        "summary": "The real -> describe -> suggest_spec -> generate_series loop.",
        "goal": "Build a matching synthetic baseline from real data.",
        "learnings": (
            "real -> spec is the key loop connecting real data and synthetic research.",
            "A spec seed does not need to be perfect; it needs to capture the main structure.",
        ),
    },
    "robust_observation_variants": {
        "title": "Same latent structure, different observation mechanisms",
        "summary": "Put irregular sampling, block missingness, measurement noise, and clipping into ObservationSpec.",
        "goal": "Evaluate robustness to observation-layer changes.",
        "learnings": (
            "Observation mechanisms should be decoupled from latent structure.",
            "Many real-world differences come from the observation layer, not the hidden dynamics.",
        ),
    },
    "external_rollout_wrap": {
        "title": "Wrap external simulation or real rollouts instead of rebuilding simulators",
        "summary": "Turn MuJoCo / PyBullet / ROS exports into TSDataForge objects.",
        "goal": "Unify external control series with the TSDataForge EDA / taskify workflow.",
        "learnings": (
            "TSDataForge owns taxonomy / EDA / taskify; mature external stacks can own simulation.",
            "That separation makes the library easier to maintain over time.",
        ),
    },
    "system_identification": {
        "title": "System identification from control input-output pairs",
        "summary": "Get past [u, y] -> future y directly.",
        "goal": "Build system-identification datasets for state-space or controlled systems.",
        "learnings": (
            "System identification is a taskification view, not a separate universe.",
            "The same base series can later become forecasting or counterfactual tasks.",
        ),
    },
    "event_control_detection": {
        "title": "Event detection for controlled time series",
        "summary": "Turn event-triggered control or bursty sequences into event-detection tasks.",
        "goal": "Generate event-detection benchmarks quickly.",
        "learnings": (
            "Trace gives event ground truth automatically, so you do not hand-label events.",
            "Control series and generic bursty event series can be compared under one task protocol.",
        ),
    },
    "causal_response": {
        "title": "Causal response: historical treatment / covariates -> future outcome",
        "summary": "A practical task for time-dependent intervention-response prediction.",
        "goal": "Generate causal-response datasets through one consistent interface.",
        "learnings": (
            "Causal and forecasting tasks can share the same TaskDataset protocol.",
            "With trace-level causal truth, you can also derive ITE and discovery tasks.",
        ),
    },
    "policy_counterfactual": {
        "title": "Policy / intervention / counterfactual paired rollouts",
        "summary": "Generate factual vs counterfactual series for the same system.",
        "goal": "Do counterfactual evaluation, policy comparison, and intervention analysis.",
        "learnings": (
            "A counterfactual pair is one of the most token-efficient units for comparison-focused agent analysis.",
            "Often you only need the factual/counterfactual difference, not every trace state.",
        ),
    },
    "policy_value_estimation": {
        "title": "Policy value estimation: from rollout to return",
        "summary": "Generate datasets with reward signals already attached.",
        "goal": "Support offline policy evaluation or return prediction.",
        "learnings": (
            "Policy workflows are not only about rollouts; they can be taskified into return targets.",
            "reward / return should live in schema instead of hidden custom code.",
        ),
    },
    "intervention_detection": {
        "title": "Intervention detection: identify when an external intervention happened",
        "summary": "Use trace intervention masks to build a detection task.",
        "goal": "Locate intervention times in control or causal systems.",
        "learnings": (
            "Interventions and events are both important supervision signals in sequences.",
            "This task is also useful for agent-driven debugging workflows.",
        ),
    },
    "real_dataset_taskify": {
        "title": "Taskify a real dataset into forecasting, classification, or causal formats",
        "summary": "Bring external arrays into one unified task protocol.",
        "goal": "Process real data and synthetic data through the same interface.",
        "learnings": (
            "TSDataForge does not require every dataset to be generated first; real datasets can enter directly.",
            "That makes agent orchestration much more stable.",
        ),
    },
    "agent_context_pack": {
        "title": "Compact context packs for agents",
        "summary": "Compress a series / dataset / task into a small, stable, structured context.",
        "goal": "Waste fewer tokens and make key information easier for agents to consume.",
        "learnings": (
            "Give agents compact schema + tags + next actions instead of raw arrays.",
            "This is more stable and more token-efficient than pasting a full README.",
        ),
    },
    "dataset_cards": {
        "title": "Automatically generate dataset and task cards",
        "summary": "Save README and JSON cards together with the dataset.",
        "goal": "Make every data asset self-explaining.",
        "learnings": (
            "Manifest + README sharply reduce follow-up communication cost.",
            "Agents can often decide how to use a dataset by reading card.json alone.",
        ),
    },
    "api_reference_overview": {
        "title": "Export the API Reference and JSON manifest in one step",
        "summary": "Turn the public API surface into a structured inventory.",
        "goal": "Ship better API docs and give agents a machine-readable manifest.",
        "learnings": (
            "API docs should be program-consumable, not only human-readable.",
            "Publishing the API manifest together with each release is agent-friendly.",
        ),
    },
    "docs_site_generation": {
        "title": "Generate a static docs / cookbook / agent-playbook site in one command",
        "summary": "Render library docs, examples, FAQ, and API into shareable HTML.",
        "goal": "Support launch, teaching, and community onboarding.",
        "learnings": (
            "Once the feature surface is coherent, docs and examples determine growth speed.",
            "Static sites are lightweight and easy to host anywhere.",
        ),
    },
    "example_routing": {
        "title": "Natural-language goal -> recommended examples -> shortest rewrite",
        "summary": "Example routing is one of the best first steps for an agent.",
        "goal": "Help new users and agents find the nearest runnable example quickly.",
        "learnings": (
            "Example routing is often more useful than a long explanation.",
            "Natural-language entry points matter a lot for adoption.",
        ),
    },
}


def _localize_example(ex: ExampleRecipe, language: str = "en") -> ExampleRecipe:
    language = (language or "en").lower()
    if language.startswith("zh"):
        return ex
    payload = _EXAMPLE_EN.get(ex.example_id)
    if not payload:
        return ex
    return ExampleRecipe(
        example_id=ex.example_id,
        title=str(payload.get("title", ex.title)),
        summary=str(payload.get("summary", ex.summary)),
        goal=str(payload.get("goal", ex.goal)),
        keywords=ex.keywords,
        audience=ex.audience,
        code=ex.code,
        learnings=tuple(payload.get("learnings", ex.learnings)),
        filename=ex.filename,
        category=ex.category,
        difficulty=ex.difficulty,
        outputs=ex.outputs,
        related_api=ex.related_api,
    )



def example_catalog(*, language: str = "en") -> list[ExampleRecipe]:
    return [_localize_example(ex, language=language) for ex in _CATALOG]



def examples_by_category(*, language: str = "en") -> dict[str, list[ExampleRecipe]]:
    grouped: dict[str, list[ExampleRecipe]] = defaultdict(list)
    for ex in example_catalog(language=language):
        grouped[ex.category].append(ex)
    return {key: list(items) for key, items in grouped.items()}



def _norm_tokens(text: str | Iterable[str]) -> set[str]:
    if isinstance(text, str):
        raw = text.lower().replace('-', ' ').replace('_', ' ')
        for ch in ',.;:/\\()[]{}\n\t':
            raw = raw.replace(ch, ' ')
        return {tok for tok in raw.split() if tok}
    out: set[str] = set()
    for item in text:
        out |= _norm_tokens(str(item))
    return out



def recommend_examples(query: str, *, top_k: int = 5, language: str = "en") -> list[ExampleRecipe]:
    """Recommend example recipes for a natural-language goal.

    The scorer is deliberately lightweight and deterministic so that it can be
    used in agent loops without extra dependencies or model calls.
    """

    q = _norm_tokens(query)
    catalog = example_catalog(language=language)
    scored: list[tuple[float, int, ExampleRecipe]] = []
    for i, ex in enumerate(catalog):
        ex_tokens = (
            _norm_tokens(ex.title)
            | _norm_tokens(ex.summary)
            | _norm_tokens(ex.goal)
            | _norm_tokens(ex.keywords)
            | _norm_tokens(ex.related_api)
            | _norm_tokens(ex.outputs)
            | _norm_tokens((ex.category, ex.difficulty))
        )
        overlap = len(q & ex_tokens)
        title_bonus = sum(1 for tok in q if tok in _norm_tokens(ex.title))
        goal_bonus = sum(1 for tok in q if tok in _norm_tokens(ex.goal))
        audience_bonus = sum(0.5 for tok in q if tok in _norm_tokens(ex.audience))
        category_bonus = 1.0 if ex.category in q else 0.0
        score = 3.0 * overlap + 1.5 * title_bonus + 1.0 * goal_bonus + audience_bonus + category_bonus
        scored.append((score, -i, ex))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = [ex for score, _, ex in scored if score > 0][: max(1, int(top_k))]
    if picked:
        return picked
    return list(catalog[: max(1, int(top_k))])



_CATALOG = _CATALOG + (
    ExampleRecipe(
        example_id="openclaw_stars_similarity",
        title="OpenClaw and GitHub star momentum: compare real-world repository attention curves",
        summary="Fetch public stargazer history, convert cumulative stars into daily momentum, and compare repo attention profiles with explainable similarity scores.",
        goal="Show how TSDataForge can turn public developer-attention data into a reproducible case study that also teaches the similarity workflow.",
        keywords=("github", "stars", "openclaw", "similarity", "attention", "live", "trend"),
        audience=("researcher", "maintainer", "agent", "community"),
        filename="openclaw_stars_similarity.py",
        category="live_similarity",
        difficulty="intermediate",
        outputs=("GeneratedSeries", "SimilarityMatrix", "EDA report"),
        related_api=("fetch_github_stars_series", "pairwise_similarity", "find_top_matches", "generate_eda_report"),
        code=dedent(
            '''
            import os
            from tsdataforge import fetch_github_stars_series, pairwise_similarity, generate_eda_report

            repos = [
                "openclaw/openclaw",
                "karpathy/llm.c",
                "microsoft/TypeScript",
            ]

            panel = {}
            for repo_id in repos:
                owner, repo = repo_id.split("/", 1)
                panel[repo_id] = fetch_github_stars_series(
                    owner,
                    repo,
                    token=os.getenv("GITHUB_TOKEN"),
                    mode="graphql",
                    since="2026-01-01",
                    max_pages=40,
                )

            matrix = pairwise_similarity(panel, difference=True, transform="zscore")
            print(matrix.to_markdown())

            generate_eda_report(
                panel["openclaw/openclaw"],
                output_path="openclaw_stars_report.html",
                title="OpenClaw star-history EDA",
            )
            '''
        ).strip(),
        learnings=(
            "Public popularity data is still time-series data: rates, shocks, surges, and decay all become explicit once you difference cumulative counts.",
            "The similarity layer helps you compare *shapes* of attention, not only absolute star totals.",
        ),
    ),
    ExampleRecipe(
        example_id="github_stars_pairwise_panel",
        title="Build a GitHub stars similarity panel for several public repos",
        summary="Create a named panel of repository star histories and rank which projects share the closest growth shape.",
        goal="Give maintainers and researchers a reusable recipe for comparing open-source attention dynamics.",
        keywords=("github", "stars", "panel", "similarity", "ranking", "open source"),
        audience=("maintainer", "researcher", "agent"),
        filename="github_stars_pairwise_panel.py",
        category="live_similarity",
        difficulty="intermediate",
        outputs=("SimilarityMatrix", "ranked matches"),
        related_api=("fetch_github_stars_series", "pairwise_similarity", "find_top_matches"),
        code=dedent(
            '''
            import os
            from tsdataforge import fetch_github_stars_series, find_top_matches, pairwise_similarity

            repos = [
                "openclaw/openclaw",
                "sindresorhus/awesome",
                "facebook/react",
                "openai/openai-python",
            ]
            panel = {}
            for repo_id in repos:
                owner, repo = repo_id.split("/", 1)
                panel[repo_id] = fetch_github_stars_series(
                    owner,
                    repo,
                    token=os.getenv("GITHUB_TOKEN"),
                    mode="graphql",
                    since="2026-01-01",
                    max_pages=40,
                )

            matrix = pairwise_similarity(panel, difference=True)
            print(matrix.to_markdown())

            reference = panel[repos[0]]
            others = {name: panel[name] for name in repos[1:]}
            matches = find_top_matches(reference, others, reference_name=repos[0], difference=True)
            for item in matches:
                print(item.to_markdown())
            '''
        ).strip(),
        learnings=(
            "A panel view makes it obvious whether two projects share similar momentum or just similar total size.",
            "The explainable score is easier to publish than an opaque embedding-only comparison.",
        ),
    ),
    ExampleRecipe(
        example_id="btc_gold_oil_similarity",
        title="BTC vs gold vs oil: compare market shapes with one reproducible workflow",
        summary="Fetch Bitcoin from CoinGecko, fetch gold and oil from FRED, then compare normalized return shapes instead of raw price levels.",
        goal="Show how TSDataForge can explain real market series and turn them into a shareable similarity case study.",
        keywords=("bitcoin", "gold", "oil", "crypto", "commodity", "similarity", "market"),
        audience=("researcher", "analyst", "agent", "community"),
        filename="btc_gold_oil_similarity.py",
        category="live_similarity",
        difficulty="intro",
        outputs=("GeneratedSeries", "SimilarityMatrix", "EDA report"),
        related_api=("fetch_coingecko_market_chart", "fetch_fred_series", "pairwise_similarity", "generate_eda_report"),
        code=dedent(
            '''
            import os
            from tsdataforge import fetch_coingecko_market_chart, fetch_fred_series, pairwise_similarity, generate_eda_report

            btc = fetch_coingecko_market_chart("bitcoin", days=180)
            gold = fetch_fred_series("GOLDPMGBD228NLBM", api_key=os.getenv("FRED_API_KEY"))
            oil = fetch_fred_series("DCOILWTICO", api_key=os.getenv("FRED_API_KEY"))

            panel = {
                "bitcoin": btc,
                "gold": gold,
                "wti_oil": oil,
            }
            matrix = pairwise_similarity(panel, log_return=True, transform="zscore")
            print(matrix.to_markdown())

            generate_eda_report(btc, output_path="bitcoin_market_report.html", title="Bitcoin market EDA")
            '''
        ).strip(),
        learnings=(
            "Comparing returns is usually more meaningful than comparing raw levels across different asset classes.",
            "One workflow can fetch, describe, compare, and publish a market case study without custom glue code.",
        ),
    ),
)

_EXAMPLE_EN.update({
    "openclaw_stars_similarity": {
        "title": "OpenClaw and GitHub star momentum: compare real-world repository attention curves",
        "summary": "Fetch public stargazer history, convert cumulative stars into daily momentum, and compare repo attention profiles with explainable similarity scores.",
        "goal": "Show how TSDataForge can turn public developer-attention data into a reproducible case study that also teaches the similarity workflow.",
        "learnings": (
            "Public popularity data is still time-series data: rates, shocks, surges, and decay all become explicit once you difference cumulative counts.",
            "The similarity layer helps you compare shapes of attention, not only absolute star totals.",
        ),
    },
    "github_stars_pairwise_panel": {
        "title": "Build a GitHub stars similarity panel for several public repos",
        "summary": "Create a named panel of repository star histories and rank which projects share the closest growth shape.",
        "goal": "Give maintainers and researchers a reusable recipe for comparing open-source attention dynamics.",
        "learnings": (
            "A panel view makes it obvious whether two projects share similar momentum or just similar total size.",
            "The explainable score is easier to publish than an opaque embedding-only comparison.",
        ),
    },
    "btc_gold_oil_similarity": {
        "title": "BTC vs gold vs oil: compare market shapes with one reproducible workflow",
        "summary": "Fetch Bitcoin from CoinGecko, fetch gold and oil from FRED, then compare normalized return shapes instead of raw price levels.",
        "goal": "Show how TSDataForge can explain real market series and turn them into a shareable similarity case study.",
        "learnings": (
            "Comparing returns is usually more meaningful than comparing raw levels across different asset classes.",
            "One workflow can fetch, describe, compare, and publish a market case study without custom glue code.",
        ),
    },
})

_CATALOG = _CATALOG + (
    ExampleRecipe(
        example_id="csv_to_report",
        title="Use your own CSV or DataFrame and get a report first",
        summary="Read your own table with pandas, call `report(...)`, and save a handoff bundle only if you want to share the result.",
        goal="Show a human-first path: read data, run one function, then open the saved report before thinking about tasks or internals.",
        keywords=("csv", "report", "handoff", "real data", "first success"),
        audience=("new_user", "researcher", "agent"),
        filename="profile_your_own_csv.py",
        category="quickstart",
        difficulty="intro",
        outputs=("report.html", "my_series_handoff/"),
        related_api=("report", "handoff", "load_asset"),
        code=dedent(
            '''
            import pandas as pd
            from tsdataforge import report, handoff

            df = pd.read_csv("my_series.csv")
            values = df[["sensor_a", "sensor_b"]].to_numpy()
            time = df["timestamp"].to_numpy()

            report(
                values,
                time=time,
                output_path="report.html",
                dataset_id="my_series",
                channel_names=["sensor_a", "sensor_b"],
            )

            bundle = handoff(
                values,
                time=time,
                output_dir="my_series_handoff",
                dataset_id="my_series",
                channel_names=["sensor_a", "sensor_b"],
            )
            print(bundle.output_dir)
            '''
        ).strip(),
        learnings=(
            "A public example should look like ordinary Python: read a table, call a function, inspect the saved result.",
            "For real data, `report(...)` is the first thing to explain, while `handoff(...)` is the follow-up for sharing and automation.",
        ),
    ),
    ExampleRecipe(
        example_id="npy_to_handoff_bundle",
        title="Use your own NumPy file and get a handoff bundle",
        summary="Point the public `report(...)` and `handoff(...)` entry points at a saved `.npy` file.",
        goal="Give new users a minimal script for real arrays without exposing the lower-level container APIs first.",
        keywords=("npy", "handoff", "bundle", "quickstart", "python"),
        audience=("new_user", "maintainer", "agent"),
        filename="npy_to_handoff_bundle.py",
        category="quickstart",
        difficulty="intro",
        outputs=("my_dataset_handoff/", "report.html"),
        related_api=("report", "handoff"),
        code=dedent(
            '''
            from tsdataforge import report, handoff

            report("my_dataset.npy", output_path="report.html")

            bundle = handoff(
                "my_dataset.npy",
                output_dir="my_dataset_handoff",
                dataset_id="my_dataset",
            )
            print(bundle.output_dir)
            '''
        ).strip(),
        learnings=(
            "The public Python surface should be as short as the CLI story: point to a file, then open the report.",
            "For humans, the first success is not a container object but a saved result they can inspect immediately.",
        ),
    ),
    ExampleRecipe(
        example_id="compare_two_dataset_versions",
        title="Compare two dataset versions and save the difference note",
        summary="Build cards and bundle assets for two dataset versions, then save a compact machine-readable diff note.",
        goal="Show that TSDataForge is useful not only for one dataset, but also for asset review across versions.",
        keywords=("dataset versions", "diff", "cards", "handoff", "qa"),
        audience=("maintainer", "researcher", "agent"),
        filename="compare_two_dataset_versions.py",
        category="launch",
        difficulty="intermediate",
        outputs=("dataset cards", "bundle", "diff note"),
        related_api=("SeriesDataset", "build_dataset_handoff_bundle", "describe_dataset"),
        code=dedent(
            '''
            from tsdataforge import SeriesDataset, build_dataset_handoff_bundle, describe_dataset
            import numpy as np

            v1 = SeriesDataset.from_arrays(np.random.default_rng(0).normal(size=(12, 192)), dataset_id="dataset_v1")
            v2 = SeriesDataset.from_arrays(np.random.default_rng(1).normal(size=(12, 192)), dataset_id="dataset_v2")

            build_dataset_handoff_bundle(v1, output_dir="dataset_v1_bundle")
            build_dataset_handoff_bundle(v2, output_dir="dataset_v2_bundle")

            d1 = describe_dataset(v1.values_list(), v1.time_list())
            d2 = describe_dataset(v2.values_list(), v2.time_list())
            print(d1.tag_counts[:5], d2.tag_counts[:5])
            '''
        ).strip(),
        learnings=(
            "Dataset report tooling becomes more valuable when it also helps review changes across asset versions.",
            "Cards and compact contexts make dataset comparisons easier to automate.",
        ),
    ),
    ExampleRecipe(
        example_id="icu_vitals_handoff_bundle",
        title="ICU / bedside vital-sign style handoff bundle",
        summary="Turn multichannel vital-sign style data into an anomaly-ready report and handoff artifact set.",
        goal="Give medicine and healthcare users a concrete dataset-report entrypoint that does not require forecasting expertise.",
        keywords=("medicine", "icu", "vitals", "anomaly", "handoff"),
        audience=("researcher", "analyst", "agent", "community"),
        filename="icu_vitals_handoff_bundle.py",
        category="eda",
        difficulty="intro",
        outputs=("report.html", "dataset card", "compact context"),
        related_api=("SeriesDataset", "build_dataset_handoff_bundle", "generate_dataset_eda_report"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import SeriesDataset, build_dataset_handoff_bundle

            t = np.arange(720, dtype=float)
            hr = 75 + 8 * np.sin(2 * np.pi * t / 120.0)
            spo2 = 98 + 0.2 * np.sin(2 * np.pi * t / 180.0)
            rr = 16 + 2 * np.sin(2 * np.pi * t / 90.0)
            values = np.stack([hr, spo2, rr], axis=1)

            ds = SeriesDataset.from_arrays([values], time=[t], dataset_id="icu_vitals_demo")
            build_dataset_handoff_bundle(ds, output_dir="icu_vitals_bundle", include_report=True)
            '''
        ).strip(),
        learnings=(
            "Cross-disciplinary adoption gets much easier once users can picture their own domain inside the report workflow.",
            "The asset layer is useful even before any clinical model is chosen.",
        ),
    ),
    ExampleRecipe(
        example_id="inflation_unemployment_regime_bundle",
        title="Inflation / unemployment regime bundle",
        summary="Build a macro-style regime report from public FRED series or from a synthetic fallback when no API key is present.",
        goal="Give economics and social-science users a concrete report + handoff workflow for regime-style time series.",
        keywords=("economics", "inflation", "unemployment", "regime", "fred"),
        audience=("researcher", "analyst", "community", "agent"),
        filename="inflation_unemployment_regime_bundle.py",
        category="eda",
        difficulty="intro",
        outputs=("report.html", "dataset card", "regime-ready handoff"),
        related_api=("fetch_fred_series", "SeriesDataset", "build_dataset_handoff_bundle"),
        code=dedent(
            '''
            import os
            from tsdataforge import fetch_fred_series, SeriesDataset, build_dataset_handoff_bundle

            api_key = os.getenv("FRED_API_KEY")
            inflation = fetch_fred_series("CPIAUCSL", api_key=api_key)
            unemployment = fetch_fred_series("UNRATE", api_key=api_key)
            ds = SeriesDataset.from_arrays(
                [inflation.values, unemployment.values],
                time=[inflation.time, unemployment.time],
                dataset_id="macro_regime_demo",
            )
            build_dataset_handoff_bundle(ds, output_dir="macro_regime_bundle", include_report=True)
            '''
        ).strip(),
        learnings=(
            "Macro series are a natural fit for the report + card + next-action story.",
            "A regime-shift discussion is easier to share once the dataset asset has explicit sidecars.",
        ),
    ),
    ExampleRecipe(
        example_id="sensor_drift_handoff_bundle",
        title="Sensor drift and change-window handoff bundle",
        summary="Create an engineering-style drift report from experimental or industrial sensor measurements.",
        goal="Show a physics / engineering entrypoint where the report layer matters before any anomaly or control model is selected.",
        keywords=("engineering", "sensor", "drift", "change point", "handoff"),
        audience=("researcher", "maintainer", "agent", "community"),
        filename="sensor_drift_handoff_bundle.py",
        category="eda",
        difficulty="intro",
        outputs=("report.html", "dataset card", "drift-ready handoff"),
        related_api=("SeriesDataset", "build_dataset_handoff_bundle", "generate_dataset_eda_report"),
        code=dedent(
            '''
            import numpy as np
            from tsdataforge import SeriesDataset, build_dataset_handoff_bundle

            t = np.arange(500, dtype=float)
            signal = 0.3 * np.sin(2 * np.pi * t / 60.0) + 0.0025 * t + (t >= 320) * 0.9
            ds = SeriesDataset.from_arrays([signal], time=[t], dataset_id="sensor_drift_demo")
            build_dataset_handoff_bundle(ds, output_dir="sensor_drift_bundle", include_report=True)
            '''
        ).strip(),
        learnings=(
            "The report layer is valuable for drift and change-window review, not only for forecasting problems.",
            "Engineering users often need a shareable artifact before they need a model benchmark.",
        ),
    ),
)
