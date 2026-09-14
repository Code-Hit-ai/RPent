Dual Franka
===========

RPent can control a two-node dual-Franka setup through an RLinf
``RealWorldEnv`` worker.

Install
-------

.. note::

	The following guide installs only the Python side (the custom RLinf
	Franka branch and ``rlinf-openpi``); it does **not** build the robot-node
	control stack the two arms need. Before installing RPent, follow the RLinf
	dual-Franka guide to set up both robot nodes: choose a compatible
	``LIBFRANKA_VERSION``, build the ``franka-franky`` (franky/libfranka) control
	stack, configure the PREEMPT_RT real-time kernel and permissions, and install
	the GELLO teleoperation and gripper dependencies. See the `RLinf dual-Franka
	guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dual_franka.html>`_.

From the RPent repository root:

.. code-block:: bash

	uv sync --extra franka

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv``.

Calibration
-----------

Hand-eye calibration is performed with ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_. It produces one YAML
per projection camera (``base_camera`` and ``d455_camera``) and saves them under
``~/.ros/easy_handeye/`` by default.

RPent reads a JSON bundle (``hand_eye_calibration.json``) that carries each
camera's ``source_name``, ``parameters``, and ``transformation``. Generate it by
copying those fields out of each ``easy_handeye`` YAML.

The bundle location is configurable with ``--calibration-path`` (default
``~/.ros/easy_handeye/hand_eye_calibration.json``).

Development configuration
-------------------------

Review and edit the checked-in development defaults before enabling motion:

* ``robots/dual_franka/config/example.yaml`` contains the machine identity (both
	robot IPs, camera serials/types, gripper connections), workspace geometry
	(target poses and safety limits), and perception localization bounds +
	base-frame transform.

RPent translates this robot-focused schema into the internal two-node RLinf
cluster and environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``.

Start the two-node Ray cluster
------------------------------

The two nodes have fixed, different roles (defined in
``robots/dual_franka/runtime_config.py``):

* Node ``0`` is the Ray head: it runs the dual-Franka environment
	worker (all cameras, perception, and arm/gripper state), and the **left**
	arm's real-time controller. For the VLA task, the
	local VLA server also runs here.
* Node ``1`` is a Ray worker: it runs only the **right** arm's real-time
	controller, with no cameras and no RPent process.

Set ``RLINF_NODE_RANK`` before starting Ray on each controller node.

Node ``0``:

.. code-block:: bash

	export RLINF_NODE_RANK=0
	ray stop --force
	ray start --head --port=6379 --node-ip-address=HEAD_IP

Node ``1``:

.. code-block:: bash

	export RLINF_NODE_RANK=1
	ray stop --force
	ray start --address=HEAD_IP:6379 --node-ip-address=WORKER_IP

Run a smoke test
----------------

Task ``0`` tests conservative single-arm analytic motion and gripper primitives:

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

RPent starts ``robots/dual_franka/env_server.py`` with the current interpreter,
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``. Task ``0`` does not load the VLA.

VLA grasp demo
--------------

RPent provides a demo that uses a VLA to grasp objects. Task ``1`` exposes
``vla_grasp`` and can start the dual-Franka VLA server locally.
``PI05_CHECKPOINT_PATH`` points to the trained Pi-05 checkpoint, while
``DUAL_FRANKA_REPO_ID`` is the dataset ID used to locate matching normalization
statistics:

.. code-block:: bash

	export PI05_CHECKPOINT_PATH=/path/to/checkpoints/global_step_N
	export DUAL_FRANKA_REPO_ID=org/dual-franka-tcp-rot6d

	uv run --extra franka rpent --robot dual_franka --task-id 1 \
	  --cuda-device 0 \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

The checkpoint must contain:

.. code-block:: text

	actor/model_state_dict/full_weights.pt
	<DUAL_FRANKA_REPO_ID>/norm_stats.json

**Pretrained checkpoint**

A ready-made task ``1`` checkpoint is published on ModelScope:
`Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000
<https://modelscope.cn/models/Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000>`_.
Download it, point ``PI05_CHECKPOINT_PATH`` at the downloaded directory, and set
``DUAL_FRANKA_REPO_ID`` to the subdirectory that holds ``norm_stats.json``:

.. code-block:: bash

	modelscope download \
	  --model Brunchlife/pi05-dualfranka-tcp-rot6d-clean-desk-532-delect-76000 \
	  --local_dir /path/to/pi05-dualfranka-clean-desk

	export PI05_CHECKPOINT_PATH=/path/to/pi05-dualfranka-clean-desk

.. warning::

	This checkpoint is trained only on our in-house test environment (robot
	poses, cameras, workspace layout, and objects), so it is expected to
	generalize poorly to a different setup. To deploy on your own rig, collect
	demonstrations and fine-tune your own checkpoint with RLinf by following the
	`RLinf dual-Franka guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dual_franka.html>`_
	(collect GELLO demos, convert to tcp_rot6d, run SFT, then deploy).

When ``--vla-endpoint`` is absent, RPent starts
``robots/dual_franka/vla_server.py`` and loads
``pi05_dualfranka_tcp_rot6d`` once.

To run the VLA service separately:

.. code-block:: bash

	uv run --extra franka python -m robots.dual_franka.vla_server \
	  --model-path /path/to/checkpoints/global_step_N \
	  --repo-id org/dual-franka-tcp-rot6d \
	  --cuda-device 0 --transport http --host 0.0.0.0 --port 6000

Then pass ``--vla-endpoint http://VLA_HOST:6000`` to ``rpent``. An external
endpoint always takes precedence over local auto-start.

External environment server
---------------------------

To attach RPent to an already-running dual-Franka environment service:

.. code-block:: bash

	uv run --extra franka rpent --robot dual_franka --task-id 0 \
	  --env-endpoint http://ROBOT_HOST:PORT \
	  --planner claude_code --model claude-opus-4-8 \
	  --robot-config robots/dual_franka/config/example.yaml \
	  --calibration-path ~/.ros/easy_handeye/hand_eye_calibration.json

Tools and artifacts
-------------------

The extension exposes ``view_env_state``, ``view_camera_meta``, ``move_delta``,
``rotate_delta``, ``open_gripper``, ``close_gripper``, and ``vla_grasp``. Each
analytic motion selects exactly one arm, ``left`` or ``right``. Mutating tools
capture per-arm state and synchronized left-wrist, base, and right-wrist images
in RPent's central ``EnvState``.

Safety
------

Keep operators at both emergency stops. Validate task ``0`` with very small
single-arm motions before attempting a grasp. Stop when camera/state results
disagree, when the requested motion is not reached, or when any calibration is
uncertain.

Attended exploration
--------------------

``dual_franka --explore`` supports the operator workflow from PR #176. It reuses
RPent's exploration sessions and layered memory while retaining the existing
real-robot RGB/depth/state logs. This does not enable single-arm ``franka``
exploration.

.. code-block:: bash

   rpent --robot dual_franka --task-id 0 --explore --interactive \
     --robot-config /path/to/robot.yaml \
     --calibration-path /path/to/hand_eye_calibration.json \
     --memory-dir /path/to/memory/dual_franka \
     --explore-attempts-per-session 3 --explore-sessions 2 \
     --output-dir /path/to/new-run

Configure the planner and task-1 VLA as described above. The client skips its
usual reset-on-connect during exploration; underlying hardware initialization
still follows RLinf's own lifecycle. Every session must call
``request_scene_reset`` before motion. The operator restores the physical scene
and replies ``done``; only a successful robot reset with camera/state capture
starts an attempt. Reset failures keep motion blocked.

``request_operator_verdict`` records a fresh observation and asks for
``success``, ``failure``, ``continue`` or ``abort``, with optional notes.
``solved()`` uses the current operator verdict. Motion and ``continue`` clear
previous verdicts. Abort/EOF permits ending without spending the remaining
attempt budget.

With ``--interactive``, reply using ``/operator <request-id> <answer>`` as shown
in the terminal; other lines remain planner steering. Without it, answer the
terminal prompt directly. A TTY is required. Dashboard operator feedback is not
implemented, so Dashboard exploration is rejected before runtime startup.

Each ``sessions/session_<NNN>/`` retains the existing artifacts plus per-step
``exploration.json`` and session-level ``operator_events.json``. Failed attempts
remain in the trace. Memory reads use ``suite`` and ``global``; working notes go
into the task inbox's ``wip/``. After success, draft suite/global lessons in the
inbox; the runner exports the winning attempt's command sequence and adds
operator evidence to its task audit. Recorded coordinates are not automatically
replayed. ``--auto-merge-memory`` is opt-in and invokes the existing memory
merge/index workflow only on successful, error-free exploration runs, including
the ``task_only`` audit/recipe pair.

Prompts are selected by ``robots/dual_franka/prompt_bundle.py``. Evaluation uses
``prompts/system.py`` and ``prompts/user.py``; exploration uses
``prompts/explore.py``. ``tasks.py`` owns task instructions, success criteria and
constraints; ``robot_spec.py`` supplies the rendering variables. Continuation
system prompts retain the task context. The original LIBERO exploration prompt
lives in ``robots/libero/prompts/explore.py``; its simulator reset/termination
assumptions are not inherited by the real robot.

External ``--env-endpoint`` servers must also be updated and advertise
``explicit_reset_only=True``; older servers are rejected before client reset.
Offline tests use fake hardware. Physical reset convergence, camera freshness
and task judgment still require validation on the deployed robot.

Direct interactive verdicts
~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``dual_franka --explore --interactive``, submit ``/success`` or ``/failure``
on its own to finish exploration through program control. Bare ``success`` and
``failure`` remain ordinary planner messages. Success requires a confirmed attempt;
failure can also end the run before the initial reset. The first terminal verdict wins.
The command never reaches the planner as chat. New tool calls are refused and
active work is cancelled at its next supported boundary; an outstanding robot
RPC or inference must return before finalization. A fresh observation backs the
operator verdict. With ``/success``, the successful recipe/audit pair is published through the
existing memory merger to ``task_only`` before exit, even without
``--auto-merge-memory``. With ``/failure``, failure evidence stays in the run directory
and no successful memory is published. Observation or persistence errors are reported as failures.
Scene reset accepts ``/done`` or ``/operator <request-id> done``. Restart the running
CLI after updating to enable this behavior.

Other operator commands
~~~~~~~~~~~~~~~~~~~~~~~

* ``/done`` confirms only the currently pending scene-reset request.
* ``/continue`` answers only the currently pending verdict request and resumes
  the attempt without marking it successful or ending the session.
* ``/abort`` cancels exploration at the supported boundary and exits, retaining
  an abort record without publishing success memory. It does not require a working camera.

Shortcuts without a matching pending request are refused, never buffered for a
future request. All five bare words without ``/`` remain ordinary agent messages.
The request-ID form ``/operator <request-id> <answer>`` remains supported.

Direct commands do not invoke a separate global/suite memory synthesis stage.
Existing planner errors remain errors and prevent automatic memory publication.

Operator diagnostic consoles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Task 103 runs explicit primitive commands without an Agent. Task 104 records
VLA inference and optional execution through the existing environment/model RPCs.
These consoles use the standard robot configuration, calibration and model flags;
no local deployment section or JX runtime is required.

.. code-block:: bash

   python -m robots.dual_franka.manual --task-id 103 \
     --robot-config /path/to/robot.yaml --calibration-path /path/to/calibration.json
   python -m robots.dual_franka.vla_test --task-id 104 \
     --robot-config /path/to/robot.yaml --calibration-path /path/to/calibration.json \
     --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset \
     --instruction "Your policy task instruction"

These are standalone module entry points; the shared ``rpent`` runner does not
dispatch tasks 103/104 to diagnostic consoles. If ``--instruction`` is omitted,
the VLA console uses the deployment task text in ``CLEAN_DESK_VLA_PROMPT``.
Override it with ``--instruction`` or the console command ``prompt <instruction>``.
Inference still requires an explicit ``infer``, ``step`` or ``run N`` command.

Manual commands include ``state``, ``cameras``, ``move right 0 0 0.01``,
``rotate right 0 0 0.05``, ``open right``, ``close right``, ``reset`` and ``quit``.
VLA commands include ``status``, ``prompt <instruction>``, ``infer`` (no motion),
``step`` (fresh inference and execution), ``run N`` (1–20 chunks), ``reset`` and
``quit``. Initialization may reset the robot. The diagnostic records predictions
before execution and rejects malformed or out-of-workspace actions. An uncertain
execution result blocks further execution until the session is restarted.

The VLA server exposes read-only ``vla.status`` metadata without changing model
precision. External VLA servers must implement this endpoint. Dual-Franka
observations must include live ``states``; missing states are rejected instead
of silently reusing a stale client cache. Records are JSON/NPZ artifacts in the
run output directory; successful RPC execution is not a task-success judgment.
