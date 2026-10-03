"""Summarize a demonstration video with the Codex SDK."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from analyse.video import extract_keyframes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frame-count', type=int, default=32)
    parser.add_argument('--max-image-edge', type=int, default=256)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--result', choices=['success', 'failure', 'unknown'], default='unknown')
    parser.add_argument('--result-note', default='')
    parser.add_argument('--codex-proxy-url', default=os.environ.get('RPENT_CONTEXT_PROXY_URL'))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error('video does not exist')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output directory must be empty to avoid overwriting results')
    output.mkdir(parents=True, exist_ok=True)
    info, frames = extract_keyframes(
        args.video.resolve(), output / 'frames', sample_stride=10,
        frame_count=args.frame_count, max_image_edge=args.max_image_edge,
    )
    import openai_codex
    from openai_codex.generated.v2_all import ReasoningEffort

    prompt = (
        'You analyze robot demonstration videos. Use only the supplied candidate images '
        'and explicitly provided outcome metadata. Do not use tools, files, networks, or other memories. '
        'Select 4 to 8 task-relevant keyframes in chronological order. '
        'Return exactly the JSON fields keyframe_indices and memory; memory must be a string.\n'
        'Write a concise English experience memory, including English titles. '
        'Divide the memory into self-contained semantic blocks. Decide the number and titles '
        'from the actual evidence; do not use fixed categories, split mechanically by frame or action, '
        'or add unsupported blocks to fill a template. Keep closely related operations together.\n'
        'Each block must start with a short title enclosed in 【】 on its own line, '
        'followed by 1 to 3 sentences. Separate blocks with exactly one blank line. '
        'Do not add a preface, conclusion, or Markdown code fences outside the blocks.\n'
        'Each block must be independently understandable. Name the objects explicitly rather '
        'than relying on pronouns or references to other blocks. Keep applicable conditions, '
        'recommendations, and related cautions in the same block so selecting it alone preserves meaning. '
        'Describe the main operation sequence and visually supported cautions using object relationships; '
        'avoid fixed coordinates, frame-by-frame narration, repetition, and excessive detail.\n'
        'Distinguish visible observations from external outcome reports. If referencing success '
        'metadata, explicitly say "The external report indicates success"; do not present it '
        'as visually verified success. Do not invent unseen release or retreat actions. '
        'Treat this single demonstration as a reference, not proof of causation, optimality, '
        'or guaranteed success.\n'
        f'Task: {args.goal}\nReported outcome: {args.result}\nOutcome note: {args.result_note}\n'
        f'Video duration: {info.duration_s:.2f}s; FPS: {info.fps}. '
        'keyframe_indices refers to candidate frame indices.'
    )
    inputs = [openai_codex.TextInput(prompt)]
    for frame in frames:
        inputs.extend([
            openai_codex.TextInput(
                f'候选帧 {frame.index}；原始帧 {frame.source_index}；时间 {frame.timestamp_s:.2f}秒'
            ),
            openai_codex.LocalImageInput(str(frame.path)),
        ])
    schema = {
        'type': 'object', 'additionalProperties': False,
        'required': ['keyframe_indices', 'memory'],
        'properties': {
            'keyframe_indices': {'type': 'array', 'items': {'type': 'integer'}},
            'memory': {'type': 'string'},
        },
    }
    overrides = {'project_doc_max_bytes': 0}
    if args.codex_proxy_url:
        proxy = args.codex_proxy_url.rstrip('/')
        overrides.update({
            'chatgpt_base_url': proxy + '/backend-api',
            'model_provider': 'rpent_context',
            'model_providers.rpent_context.name': 'rpent_context',
            'model_providers.rpent_context.base_url': proxy + '/backend-api/codex',
            'model_providers.rpent_context.requires_openai_auth': True,
            'model_providers.rpent_context.supports_websockets': False,
        })
    config = openai_codex.CodexConfig(
        cwd=str(output), env=dict(os.environ), experimental_api=True,
        config_overrides=tuple(f'{k}={json.dumps(v)}' for k, v in overrides.items()),
    )
    print(f'Analyzing {len(frames)} candidate frames with {args.model}', flush=True)
    with openai_codex.Codex(config=config) as codex:
        thread = codex.thread_start(
            model=args.model, sandbox=openai_codex.Sandbox.read_only,
            approval_mode=openai_codex.ApprovalMode.deny_all,
            developer_instructions='Analyze supplied images only. Do not invoke tools.',
        )
        result = thread.turn(inputs, effort=ReasoningEffort.low, output_schema=schema).run()
    if getattr(result.status, 'value', result.status) != 'completed':
        raise RuntimeError(f'Codex analysis failed: {result.status}: {result.error}')
    payload = json.loads(result.final_response)
    indices = payload['keyframe_indices']
    if not 4 <= len(indices) <= 8 or any(type(i) is not int or i not in range(len(frames)) for i in indices):
        raise ValueError('model returned invalid keyframe indices')
    if len(set(indices)) != len(indices) or not isinstance(payload['memory'], str) or not payload['memory'].strip():
        raise ValueError('model returned duplicate frames or empty memory')
    # Validate the same block contract consumed by prompt injection and Memory NC.
    from rpent.context.memory import load_memory_blocks

    memory_path = output / 'memory.txt'
    memory_path.write_text(payload['memory'].strip() + '\n', encoding='utf-8')
    load_memory_blocks(memory_path)
    selected = [frames[i] for i in sorted(indices)]
    (output / 'keyframes').mkdir()
    for frame in selected:
        shutil.copy2(frame.path, output / 'keyframes' / frame.path.name)
    manifest = {
        'video': str(args.video.resolve()), 'goal': args.goal, 'model': args.model,
        'reasoning_effort': 'low', 'reported_result': args.result,
        'result_note': args.result_note, 'candidate_count': len(frames),
        'selected_frames': [f.as_dict() for f in selected],
    }
    (output / 'selection.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(payload['memory'], flush=True)
    print(f'Saved {len(selected)} keyframes and memory to {output}', flush=True)


if __name__ == '__main__':
    main()
