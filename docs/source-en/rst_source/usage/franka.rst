Franka
======

RPent can control one physical Franka arm through an RLinf ``RealWorldEnv``
worker.

Install
-------

.. note::

	The following guide installs only the Python side (the custom RLinf
	Franka branch and ``rlinf-openpi``); it does **not** build the robot
	controller stack the arm needs. Before installing RPent, follow the RLinf
	single-arm Franka guide to set up the controller node: check Franka firmware
	compatibility, install the real-time kernel, choose your gripper (Franka hand
	or Robotiq 2F-85/2F-140) and camera, and build the ROS control packages (ROS
	Noetic, the matching libfranka and franka_ros, and serl_franka_controllers).
	See the `RLinf single-arm Franka guide
	<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka.html>`_.

From the RPent repository root:

.. code-block:: bash

   uv sync --extra franka

This installs the custom RLinf Franka branch and ``rlinf-openpi`` into
``.venv``.

Calibration
-----------

Hand-eye calibration is performed with ROS
`easy_handeye <https://github.com/IFL-CAMP/easy_handeye>`_. It produces one YAML
per camera (eye-on-base for the external camera, eye-on-hand for the wrist
camera) and saves them under ``~/.ros/easy_handeye/`` by default.

RPent loads those YAMLs directly: list them under ``perception.calibration`` in
the robot config, mapping each camera to its easy_handeye YAML (the checked-in
``robots/franka/config/example.yaml`` already does this):

.. code-block:: yaml

   perception:
     calibration:
       external: ~/.ros/easy_handeye/fr3_external_apriltag_eye_on_base.yaml
       wrist: ~/.ros/easy_handeye/fr3_wrist_apriltag_ee_eye_on_hand.yaml

Paths may be absolute, ``~``-prefixed, or relative; relative paths resolve
against the working directory RPent is launched from.

Development configuration
-------------------------

The checked-in values are development defaults and must be reviewed before
enabling motion:

* ``robots/franka/config/example.yaml`` contains the machine identity (robot IP,
	camera serials, gripper), workspace geometry (target/reset poses and safety
	limits), and the easy_handeye YAML mapping (see Calibration).

RPent translates this robot-focused schema into the internal RLinf cluster and
environment objects. To use a different file, pass
``--robot-config /path/to/robot_config.yaml``.

Start Ray
---------

Set the node rank before starting Ray, because Ray captures the environment at
startup:

.. code-block:: bash

   export RLINF_NODE_RANK=0
   ray stop --force
   ray start --head

Run a smoke test
----------------

The smoke test verifies that basic analytic motion and gripper primitives work
correctly. To run it, launch RPent with task ``0``:

.. code-block:: bash

   # replace --robot-config with your own config
   uv run --extra franka rpent --robot franka --task-id 0 \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/franka/config/example.yaml

RPent starts ``robots/franka/env_server.py`` with the current interpreter,
loads the RPent robot config, generates the internal RLinf adapter config,
connects to Ray, waits for ``healthz``, and records the initial state as step
``0``.

VLA grasp demo
--------------

RPent provides a demo that uses a VLA to grasp objects. Task ``1`` exposes
``vla_grasp``. Single Franka currently requires a compatible external VLA
service whose observation layout, action layout, checkpoint, and normalization
statistics match the current Franka training configuration:

.. code-block:: bash

   uv run --extra franka rpent --robot franka --task-id 1 \
     --vla-endpoint http://VLA_HOST:PORT \
     --planner claude_code --model claude-opus-4-8 \
     --robot-config robots/franka/config/example.yaml

The VLA server must be deployed separately for now. Without
``--vla-endpoint``, analytic motion and gripper tools remain available, but
``vla_grasp`` raises a runtime error.

Tools and artifacts
-------------------

The extension exposes ``view_env_state``, ``view_camera_meta``, ``move_delta``,
``rotate_delta``, ``open_gripper``, ``close_gripper``, and ``vla_grasp``.
Mutating tools capture robot state, wrist and external RGB images, optional
aligned depth arrays, and camera metadata in RPent's central ``EnvState``.

Safety
------

Keep an operator at the emergency stop. Validate task ``0`` with very small
motions before attempting a grasp. Stop when camera/state results disagree,
when the requested motion is not reached, or when any calibration is uncertain.


.. _franka-flash:

Franka Flash plans (task cards)
================================

Single and dual Franka support ``--planner flash`` using a reviewed version-1
JSON plan. A plan records primitive order and motion intent; Molmo selects a
pixel in a fresh camera image for each translation. Existing calibrated depth
projection converts it into robot coordinates. No planning model is called.

The stored translation offset is the demonstrated **actual TCP endpoint**
minus the demonstrated anchor position. Replay adds this offset to the newly
localized anchor and subtracts the current TCP position to obtain the move.
Action order is fixed; interpolation remains the existing controller's responsibility.

First configure the robot and ``perception.calibration`` using the setup above,
start a Molmo service, and record a successful run. Review ``states.json`` and
write ``annotations.json`` keyed by the source step number::

   {
     "1": {"intent": "approach the cup rim", "phrase": "visible cup rim", "camera": "third_person"},
     "2": {"intent": "align the gripper", "rotation_mode": "relative"}
   }

Every translation needs a semantic anchor; every rotation needs an intent and
``relative`` or ``fixed`` rotation mode. Single Franka uses ``third_person`` or
``wrist``; dual Franka uses ``base``, ``d455``, ``left_wrist`` or ``right_wrist``.
The selected view must have depth and valid calibration. Keep each recorded
translation within 0.20 m and each rotation within 0.35 rad.

Generate a plan offline from the reviewed recording (Molmo must be reachable)::

   python -m robots.franka.flash.generate \
     --robot franka --task franka_t0 \
     --robot-config /path/to/robot.yaml \
     --run-dir /path/to/successful-run --annotations annotations.json \
     --molmo-endpoint http://localhost:9000 --destination plan.json

Confirm the source success when prompted. The generator rejects unsupported
commands and does not overwrite an existing destination. It stores hashes of
the robot configuration and calibration files; regenerate and review the plan
if those files change. Dual Franka uses ``--robot dual_franka --task dual_franka_t0``.

Replay from a dedicated operator terminal::

   rpent --robot franka --planner flash --task-id 0 \
     --robot-config /path/to/robot.yaml --flash-plan plan.json \
     --molmo-endpoint http://localhost:9000

For dual Franka, use ``--robot dual_franka`` and the corresponding plan/config.
Configure Env/VLA endpoints as in the robot setup guide; single-arm plans with
``vla_grasp`` require ``--vla-endpoint``. Replay asks before driver initialization
(which may reset the robot), before execution, and for the final task verdict.
Do not use ``--interactive``, ``--dashboard`` or ``--explore`` with this mode.

Invalid pixels/depth, workspace violations, excessive motions, and primitive
errors stop replay. Left-arm workspace bounds are checked in the left-base
frame. A successful RPC does not certify task success: the final operator
verdict determines the result. ``flash_outcome.json`` and
``flash_recipe.jsonl`` record the outcome and issued actions.

Only reviewed v1 plans are supported. Experimental supervised v2 plans,
stage splicing, reference-image selection and historical-point reuse are not
part of this interface. These commands require robot-specific installation and
operator validation; offline unit tests do not establish real-robot success.

Optional agent grounding fallback
------------------------------------------

Add ``--grounding-agent-model provider:model`` to the replay command to enable
one fallback attempt after Molmo fails. ``codex:model`` uses the existing Codex
CLI login/provider configuration; API models use the existing API model factory.
``--grounding-agent-base-url`` optionally overrides the model endpoint.
Use a model that accepts images and returns structured output.

For both single and dual Franka, a missing target, invalid pixel/depth or Molmo
request failure triggers a fresh observation and one agent selection on the
same named object part. The agent only selects pixels and receives no robot
tools. Its point must pass the same depth projection and workspace/motion checks.
If it fails, replay stops without executing that motion. The next translation
starts with Molmo again. Cancellation does not trigger fallback. Configuration
errors and workspace/motion-limit failures stop directly.

The fallback is disabled by default and applies to live replay, not offline
plan generation. Each attempt is recorded in per-step ``flash_grounding.json``
artifacts. Agent calls can incur model costs and have a 90-second request timeout;
Flash planner token counters do not include these perception requests.
