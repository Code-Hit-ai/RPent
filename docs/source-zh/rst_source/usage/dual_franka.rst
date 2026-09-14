Dual Franka
===========

RPent 可以通过 RLinf ``RealWorldEnv`` worker 控制双节点双臂 Franka 系统。

安装
----

.. note::

	以下的步骤只会安装 Python 侧依赖（自定义的 RLinf Franka 分支和
	``rlinf-openpi``），并 **不会** 构建双臂真正需要的机器人节点控制栈。在安装
	RPent 之前，请先按照 RLinf 双臂 Franka 指南配置两个机器人节点：选择兼容的
	``LIBFRANKA_VERSION``，构建 ``franka-franky`` （franky/libfranka）控制栈，配置
	PREEMPT_RT 实时内核与相关权限，并安装 GELLO 遥操作与夹爪依赖。参见
	`RLinf 双臂 Franka 指南
	<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_。

在 RPent 仓库根目录运行：

.. code-block:: bash

	uv sync --extra franka

该命令将自定义 RLinf Franka 分支和 ``rlinf-openpi`` 安装到 ``.venv``。

标定（Calibration）
----------------------

手眼标定使用 ROS `easy_handeye
<https://github.com/IFL-CAMP/easy_handeye>`_ 完成。它为每台投影相机（
``base_camera`` 和 ``d455_camera``）生成一个 YAML，默认保存在
``~/.ros/easy_handeye/`` 下。

RPent 会读取一个 JSON 文件（``hand_eye_calibration.json``），其中包含每台相机的
``source_name``、``parameters`` 和 ``transformation``。生成方式是从每个
``easy_handeye`` YAML 中复制这些字段。

该文件的位置可通过 ``--calibration-path`` 配置（默认
``~/.ros/easy_handeye/hand_eye_calibration.json``）。

开发配置
--------

启用机械臂运动前，请检查并修改仓库中的开发默认值：

* ``robots/dual_franka/config/example.yaml`` 包含机器人身份（两台机器人 IP、相机
  序列号/类型、夹爪连接）、工作空间几何（目标位姿、安全边界）和感知定位
  边界 + base-frame 变换。

RPent 会将该机器人配置转换成内部双节点 RLinf cluster 和环境对象。如需使用
其他文件，请传入 ``--robot-config /path/to/robot_config.yaml``。

启动双节点 Ray 集群
--------------------

两个节点的角色不同（定义在 ``robots/dual_franka/runtime_config.py``
中）：

* 节点 ``0`` 是 Ray head 节点：运行双臂 Franka 环境 worker
  （全部相机、感知以及双臂和夹爪状态）和**左臂**的实时控制器。VLA 任务的
  本地 VLA 服务也运行在该节点上。
* 节点 ``1`` 是 Ray worker 节点：只运行**右臂**的实时控制器，不接相机，
  也不运行 RPent 进程。

每个控制节点都必须在启动 Ray 前设置 ``RLINF_NODE_RANK``。

节点 ``0``：

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head --port=6379 --node-ip-address=HEAD_IP

节点 ``1``：

.. code-block:: bash

	export RLINF_NODE_RANK=1
	ray stop --force
	ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

运行冒烟测试
------------

任务 ``0`` 用于测试稳妥的单臂解析式运动和夹爪 primitives：

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent 使用当前解释器启动 ``robots/dual_franka/env_server.py``，加载 RPent
robot config 并生成内部 RLinf adapter config，然后连接 Ray，等待 ``healthz``，
并将初始状态记录为 step ``0``。任务 ``0`` 不会加载 VLA。

VLA 抓取 DEMO
-------------

RPent 提供了一个使用 VLA 抓取物品的 DEMO。task-id ``1`` 会暴露 ``vla_grasp``，
并可在本地启动双臂 Franka VLA 服务。``PI05_CHECKPOINT_PATH`` 指向 
训练好的 Pi-05 checkpoint，``DUAL_FRANKA_REPO_ID`` 是用于查找对应归一化统计的数据集 ID：

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --robot dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

checkpoint 必须包含：

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

**预训练 checkpoint**

ModelScope 上发布了一个可直接使用的 task ``1`` checkpoint：
`Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000
<https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_。
下载后将 ``PI05_CHECKPOINT_PATH`` 指向下载目录，并将 ``DUAL_FRANKA_REPO_ID``
设置为包含 ``norm_stats.json`` 的子目录：

.. code-block:: bash

	modelscope download \
	  --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
	  --local_dir /path/to/pi05-dualfranka-clean-desk

	export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk

.. warning::

	该 checkpoint 仅在我们的内部测试环境（机器人位姿、相机、工作空间布局和物体）
	上训练，切换到不同的环境时预计表现会较差。若要部署到你自己的机器上，请使用
	RLinf 采集示教数据并微调你自己的 checkpoint，参见
	`RLinf 双臂 Franka 指南
	<https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/dual_franka.html>`_
	（采集 GELLO 示教数据、转换为 tcp_rot6d、运行 SFT，然后部署）。

未设置 ``--vla-endpoint`` 时，RPent 会启动
``robots/dual_franka/vla_server.py``，并只加载一次
``pi05_dualfranka_tcp_rot6d``。

也可以单独启动 VLA 服务：

.. code-block:: bash

	uv run --extra franka python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

然后向 ``rpent`` 传入 ``--vla-endpoint http://VLA_HOST:6000``。外部 endpoint
始终优先于本地自动启动。

连接外部环境服务
----------------

连接已经运行的双臂 Franka 环境服务：

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

工具与状态产物
--------------

双臂 Franka 扩展提供 ``view_env_state``、``view_camera_meta``、
``move_delta``、``rotate_delta``、``open_gripper``、``close_gripper`` 和
``vla_grasp``。每次解析式运动只会作用于一条臂（``left`` 或 ``right``）。
所有会改变环境状态的工具都会在 RPent 统一的 ``EnvState`` 中保存每条臂的状态以及
同步的 left-wrist、base 和 right-wrist 图像。

安全要求
--------

两条臂都必须有操作员留在急停按钮旁。先使用极小的单臂动作验证任务 ``0``，
再尝试抓取。当相机与状态结果不一致、目标运动没有到位，或任何标定存在疑问时，
应立即停止。

探索模式（人工复位与判定）
--------------------------

``dual_franka`` 支持基于 PR #176 操作员接口的 ``--explore``。使用原来的真机
状态和相机日志，通过已有的探索循环执行多次尝试、跨 session 交接及 memory 整理。
单臂 ``franka`` 尚未开放该模式。

在已配置好的机器人运行环境中启动，例如使用任务 0：

.. code-block:: bash

   rpent --robot dual_franka --task-id 0 --explore --interactive \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --memory-dir /path/to/memory/dual_franka \
     --explore-attempts-per-session 3 --explore-sessions 2 \
     --output-dir /path/to/new-run

规划器参数和 VLA 配置仍按前文设置；任务 1 的 checkpoint/外部 VLA 服务要求不变。
每个 session 的第一轮也需要确认场景，连接环境客户端时不会额外调用复位。
这不替代底层 RLinf/机器人控制器自身的启动流程。

* ``request_scene_reset(reason, expected_scene_state)``：操作员处理夹持物体并恢复
  桌面后输入 ``done``，工具再调用机器人 ``reset()``。只有复位返回成功且取得
  配置要求的相机观测与双臂状态后，才开始一个新的 attempt；失败时保持运动关闭。
* ``request_operator_verdict(question)``：记录新观测后，操作员回复
  ``success``、``failure``、``continue`` 或 ``abort``，可附备注。
  ``solved()`` 只读取当前有效的人工成功判定。后续运动或 ``continue`` 会清除旧判定。
* ``abort`` 或操作员输入端关闭会终止本次探索；不会为了耗尽预算强迫继续。

交互模式下，回复使用终端显示的请求 ID，例如 ``/operator <id> done`` 或
``/operator <id> success 物体已稳定提起``。普通输入仍用于指导 agent；只有匹配
当前请求 ID 的回复才用于操作员确认，避免与 ``--interactive`` 争抢 stdin。
不加 ``--interactive`` 时直接按提示输入答案。两种方式均要求 TTY；目前未实现
Dashboard 操作员反馈，``--dashboard --explore`` 会在启动服务前报错。

日志与 memory
~~~~~~~~~~~~~

每个 session 的 ``sessions/session_<NNN>/`` 保留原有 ``states.json``、左右腕部、
base、可用的 D455 图像/深度和相机元数据。复位不清空前一次尝试的记录。
新增 ``exploration.json`` 步骤附件记录 attempt 边界，``operator_events.json``
记录人工反馈和对应的观测步骤。

探索沿用 RPent 的 memory 管理器：

* 读取任务的 ``suite`` 经验和 ``MEMORY.md`` / ``global`` 通用经验。
* 失败证据写到运行目录的 ``attempts/``，工作笔记写到
  ``<memory-dir>/_internal/inbox/<recipe_tag>/wip/``。
* 人工确认成功后，整理 inbox 中的 ``suite`` / ``global`` 草稿。
  runner 从最后一次成功复位后的执行记录导出 recipe，并为任务 audit 补入人工判定
  及状态证据。recipe 保存实际调用序列，不自动执行旧坐标。
* 默认不自动合并。加 ``--auto-merge-memory`` 后，只在本次运行成功且无 agent 错误时，
  使用现有 merge/validate/index 流程发布草稿和 ``task_only`` audit/recipe 对。
  失败或中止的运行不自动发布草稿；原始日志与工作笔记继续保留。

提示词维护
~~~~~~~~~~

``robots/libero/prompts/explore.py`` 是原有 LIBERO 探索提示词；它包含仿真复位和
``libero_terminated`` 约定，不能直接用于真机。

双臂的 ``robots/dual_franka/prompt_bundle.py`` 根据 ``mode`` 选择：普通运行由
``prompts/system.py`` 和 ``prompts/user.py`` 组装；探索运行使用
``prompts/explore.py``。任务名称、指令、成功标准和约束来自 ``tasks.py``，由
``robot_spec.py`` 填充变量，统一通过 ``PromptBundle.render()`` 渲染。
跨 session 的 system prompt 也包含当前任务及成功标准。

真机探索提示词保留双臂工具和坐标约定，单独定义人工复位/判定、失败归档及
三层 memory 流程；不继承 LIBERO 的“任务保证可解”或自动恢复物体等仿真假设。

使用 ``--env-endpoint`` 连接外部服务时，服务也必须更新到本实现，并在环境元数据中
声明 ``explicit_reset_only=True``；旧服务会在客户端复位前被拒绝，避免保留其自动
复位行为。探索回归仅使用假硬件，实际相机新鲜度、机械臂复位到位和任务判定仍需
在部署现场验收。

交互式结束指令
~~~~~~~~~~~~~~

在双臂 ``--explore --interactive`` 模式中，单独输入 ``/success`` 或 ``/failure`` 并回车，由程序直接结束当前探索，
不作为聊天消息交给规划器。不带斜杠的 ``success``、``failure`` 和其他自然
语言照常交给 Agent 理解。
``/success`` 仅在场景确认并成功复位之后接受；``/failure`` 也可在等待
首次复位时结束。两者都不需要等待 Agent 请求结果判定。首次接受的结束判定
不会被后续相反判定覆盖。

程序立即禁止新工具调用，并请求取消执行中的工具。VLA 在现有动作边界检查
取消；已经发送的机器人动作、正在执行的 RPC 或推理需要返回后才能收尾。
随后采集新观测并记录人工判定。``/success`` 导出本次成功尝试的动作序列和审计证据，
调用现有 memory 合并接口保存到 ``task_only``，然后关闭服务并退出。
``/failure`` 保留运行日志和失败判定并退出，不发布成功经验。
``/success`` 自动执行 memory 合并，无需额外指定 ``--auto-merge-memory``。
观测或保存失败时不会报告完整成功，会返回错误；运行目录保留诊断记录。

包含其他文字的消息（例如“抓取 success，但整个任务还没完成”）仍是普通
聊天消息。场景重置可直接回复 ``/done``，也兼容 ``/operator <request-id> done``。
代码更新需要重启当前任务进程后生效。

其他程序级操作员指令
~~~~~~~~~~~~~~~~~~~~

* ``/done``：确认当前等待的场景准备请求，允许复位。不等待复位确认时会拒绝，
  不会提前缓存并触发之后的复位。
* ``/continue``：回复当前结果判定请求，继续本轮尝试；不结束、不标记成功。
  没有结果判定请求时会提示拒绝。
* ``/abort``：中止探索，禁止后续工具调用，在现有取消边界收尾后退出。
  保留中止记录，不发布成功 memory；相机不可用也可以中止。

五个指令都需要单独输入并回车。不带 ``/`` 的 ``done``、``success``、
``failure``、``continue``、``abort`` 均属于普通聊天，不控制程序。

直接指令不会额外调用 global/suite 经验提炼。已有规划器错误仍会保留并阻止自动发布 memory。

操作员诊断控制台
~~~~~~~~~~~~~~~~

任务 103 无需 Agent，逐条执行原语命令；任务 104 通过现有环境和模型 RPC
记录 VLA 推理及可选执行。使用标准机器人配置、标定和模型参数，不依赖本机
deployment 配置或 JX 运行时。

.. code-block:: bash

   python -m robots.dual_franka.manual --task-id 103 \
     --robot-config /path/to/robot.yaml --calibration-path /path/to/calibration.json
   python -m robots.dual_franka.vla_test --task-id 104 \
     --robot-config /path/to/robot.yaml --calibration-path /path/to/calibration.json \
     --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset \
     --instruction "完整策略任务指令"

这两个控制台使用独立模块入口；共享 ``rpent`` runner 不再将任务 103/104
分发到诊断控制台。未提供 ``--instruction`` 时，VLA 控制台使用
``CLEAN_DESK_VLA_PROMPT`` 中的部署任务文字，可通过 ``--instruction`` 或
控制台的 ``prompt <指令>`` 覆盖。仍需显式输入 ``infer``、``step`` 或
``run N`` 才触发推理。

手动命令包括 ``state``、``cameras``、``move right 0 0 0.01``、
``rotate right 0 0 0.05``、``open right``、``close right``、``reset``、``quit``。
VLA 命令包括 ``status``、``prompt <指令>``、``infer``（仅推理）、
``step``（重新观察推理并执行）、``run N``（1–20 块）、``reset``、``quit``。
初始化可能复位。诊断在执行前保存预测，拒绝格式错误或超出工作空间的动作；
执行结果不确定时禁止继续执行，需重启会话。

VLA 服务增加只读 ``vla.status`` 元数据查询，不改变模型精度；外部 VLA 服务
需要支持该接口。双臂观测必须返回实时 ``states``，缺失时明确报错，不再静默
使用旧缓存。运行目录保存 JSON/NPZ 记录；RPC 成功不表示任务成功。
