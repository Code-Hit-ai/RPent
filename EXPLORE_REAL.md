# 真机探索独立工作目录

本目录从上游 RPent 独立 clone，包含 main、完整 PR #176，以及双臂真机探索管线接入。未从旧目录复制 V2 功能、数据、checkpoint 或环境。

## 固定基线

- 上游： https://github.com/RLinf/RPent
- 分支：`explore-real`
- main：`e6c585ddabfaf48adb6e5538e9b5a325cb9e88b9`
- PR #176：`528334c06884391e600c7d149e624c7bb77a2013`
- main + PR #176 的本地合并提交：`5cc475191831621c4b6ab144f3eefb932aab79d8`

探索接入作为该合并提交之后的独立提交。审查接入部分可运行：

```bash
git diff 5cc475191831621c4b6ab144f3eefb932aab79d8..HEAD
```

相对于 main 的 PR 会同时包含 PR #176 和探索接入改动。未向远程推送或创建 PR。

## 接入范围

- `dual_franka --explore` 复用探索 session、attempt 和已有 memory 流程。
- 连接时不自动 reset；每个 session 从人工场景确认开始。reset 成功且记录配置要求的相机和双臂状态后才允许运动。
- `request_operator_verdict` 采集当前观测后请求人工判定，`solved()` 读取有效判定；后续运动会清除成功判定。
- 保留 PR #176 的具名 VLA、关节恢复、投影和相机配置；为运动与旧 attempt 的感知引用增加探索状态检查。
- 保留真机原有状态/相机日志，补充 attempt 和操作员证据。memory 使用 main 原有的 global/suite/task_only、inbox、merge/validate/index 流程。
- 默认不发布 memory 草稿；显式 `--auto-merge-memory` 仅在成功且无 agent 错误时自动合并。
- CLI 操作员输入按请求 ID 路由，支持中止与 EOF；当前需要 TTY，未开放 Dashboard 和单臂 franka 探索。

启动参数、日志和 prompt 位置见 [双臂真机文档](docs/source-zh/rst_source/usage/dual_franka.rst) 的“探索模式”部分。需要配置机器人、标定、规划器及对应 VLA 服务。外部环境服务也需使用本实现，提供 `explicit_reset_only=True` 元数据。

## 验证记录

- 离线单元测试：444 passed，2 skipped。
- 完整收集因测试环境缺少 `torch` 失败；上述结果排除了 `tests/unit_tests/robots/robotwin/test_seed_language_contract.py`。
- Ruff 检查与 `git diff --check` 通过。
- 测试使用假硬件，覆盖人工 reset/verdict、运动门控、中止、跨 session、日志、memory、输入路由，以及启动时读取观测不调用 reset/step。
- 未启动机器人或模型服务，未做真机验收。

本次复用已有外部 Python 环境执行测试，未复制或安装环境：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
PYTHONPATH="$PWD" RPENT_REPO_ROOT="$PWD" \
/tmp/rpent-agent-vision-dTT8GE/venv/bin/python -m pytest tests/unit_tests \
  --ignore=tests/unit_tests/robots/robotwin/test_seed_language_contract.py \
  --disable-warnings --tb=short -q --junitxml=logs/explore-offline/junit.xml
```

原目录 `/mnt/public2/zhangyixian/RPent0910_Full` 保持不变：此前 222 个修改/未跟踪文件的 SHA256 与文件列表均已核对。
