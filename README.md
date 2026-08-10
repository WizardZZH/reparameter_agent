# Failure-Driven Parametric Model Synthesis

根据 `method.txt` 实现的参数模型合成流程。每个任务处理一个 PM 目录：用训练集迭代合成 Python `generate_*` 函数，再用独立测试集报告泛化覆盖率。

## 入口

唯一入口脚本是 `synthesize_pm.py`。目标运行环境为 `/opt/conda/envs/agent/bin/python`（包含 AgentScope 1.0.17 和 PyYAML）。

```bash
/opt/conda/envs/agent/bin/python synthesize_pm.py \
  --dataset-dir /data-nas/data/experiments/genyue/code/genyue/reparameterCAD/dataset/58794631
```

常用可选参数：

- `--batch-size 5`：每轮从当前训练失败集抽取的样本数；
- `--max-rounds 10`：最大迭代数；
- `--seed 42`：失败集采样种子；
- `--tolerance 0.01`：panel 坐标绝对误差，单位 mm；
- `--config agent/config.yaml`：AgentScope 模型配置。

## 数据目录

传入目录名必须是 PM ID，并固定包含：

```text
dataset/58794631/
├── input_200.json  # Failure-Driven 合成与 Evidence Gate
└── test_50.json    # 最终模型确定后，仅评估一次
```

每个文件是样本数组。每个样本必须包含：

- `sample_id`
- `pm_id`，必须等于目录名
- `category`
- `paramModelId`
- `input_pm`，每项包含 `name`、`value`、`type`
- `panels`，每项为 `[x1, y1, z1, x2, y2, z2]`

所有 train/test 样本必须具有相同 PM、category、paramModelId、参数顺序和参数类型，且 sample ID 不能重复。

## 迭代与覆盖率

`input_200.json` 是唯一优化数据：每轮依次执行 Domain Agent、Coding Agent、全训练集确定性验证、Evidence Gate 和 Verifier Agent。候选仅在保留全部历史成功样本且训练覆盖严格提高时提交。

`test_50.json` 不会发送给 Agent，不参与采样、提示、Evidence Gate 或模型选择。最终 PM 提交后才执行一次 held-out 测试。

每次运行都会创建不可覆盖的目录：

```text
output/<pm-id>_<UTC时间>_<UUID>/
```

主要结果：

- `final/model.py`：最终参数模型；
- `final/evaluation.json`：最终训练集验证；
- `final/test-evaluation.json`：独立测试集验证；
- `final/summary.json`：`train_coverage`、`test_coverage`、pass/fail IDs 和迭代轨迹；
- `rounds/`：每轮候选、对话、差异、gate 和日志；
- `input/`：原始与标准化 train/test 输入。

退出码含义：

| 退出码 | 状态 | 含义 |
| --- | --- | --- |
| `0` | `success` | 训练集达到 100% 覆盖，流程正常成功结束；`0` 不是报错。 |
| `2` | `partial` | 未达到训练集全覆盖，但已经保留并输出当前最佳的部分覆盖模型。 |
| `1` | `failed` / `fatal-error` | 数据或环境异常、流程发生致命错误，或者没有找到可提交的有效改进模型。 |

程序结束时会输出类似 `{"exit_code": 0, "workspace": "output/..."}` 的 JSON：`exit_code` 表示上述执行结果，`workspace` 是本次运行的结果目录。具体终止原因可查看该目录下的 `final/summary.json`（`termination_reason`）；若发生致命错误，则查看 `failure.json` 中的 `error` 和 `traceback`。

退出码只由训练集上的合成结果决定。`test_50.json` 的测试覆盖率仅用于报告，不会改变退出码。

## 外部 Coordinator 调用 Sub-agent

主流程以相同 CLI 调用三个 builder-style sub-agent：

```bash
/opt/conda/envs/agent/bin/python -m pm_synthesis.agent_cli \
  --role domain \
  --workspace output/<task-directory> \
  --round 1 \
  --config agent/config.yaml
```

`--role` 可取 `domain`、`coding`、`verifier`。请求和结果位于对应轮次的 `agents/<role>/` 下。

## 测试

```bash
/opt/conda/envs/agent/bin/python -m unittest discover -s tests -v
```

付费真实模型 smoke test 默认跳过；显式启用方式：

```bash
RUN_LIVE_PM_SYNTHESIS=1 \
PM_SYNTHESIS_LIVE_DATASET_DIR=/path/to/small-pm-directory \
/opt/conda/envs/agent/bin/python -m unittest tests.test_live_smoke -v
```
