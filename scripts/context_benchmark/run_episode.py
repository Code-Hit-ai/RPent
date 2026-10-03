import base64,hashlib,json,os,sys,time
from pathlib import Path
from rpent.tools.toolkit import ToolResult
from rpent.planner.codex import CodexPlanner
from threading import Lock
clock_lock=Lock()
clock_data={}
original_solve=CodexPlanner.solve

def write_clock():
    (out/'timing.json').write_text(json.dumps(clock_data,ensure_ascii=False,indent=2))

def mark_end(reason):
    with clock_lock:
        if 'started_monotonic' in clock_data and 'task_duration_s' not in clock_data:
            clock_data['task_duration_s']=time.perf_counter()-clock_data['started_monotonic']
            clock_data['task_end_at']=time.time()
            clock_data['task_end_reason']=reason
            write_clock()

def solve(self,*args,**kwargs):
    compact = '--prompt-profile=compact' in sys.argv or any(
        flag == '--prompt-profile' and value == 'compact'
        for flag, value in zip(sys.argv, sys.argv[1:])
    )
    if not compact:
        kwargs["user_message"] += (
            "\n\nFor this run, first discover and call view_env_state to read the current "
            "task_language and initial observation before reading task memories or guides. "
            "This read-only observation does not authorize motion or reset. Then read the "
            "required guides and relevant memories before any manipulation. Current task_language "
            "is authoritative; task numbers in other suites do not identify this task. "
            "If the exact task reference is absent, skip it and use memories relevant to "
            "the observed task; do not substitute an unrelated task from another suite."
        )
    clock_data.update(started_monotonic=time.perf_counter(),started_at=time.time())
    write_clock()
    try:
        return original_solve(self,*args,**kwargs)
    finally:
        mark_end('planner_ended_without_earlier_task_verdict')
        with clock_lock:
            clock_data['codex_duration_s']=time.perf_counter()-clock_data['started_monotonic']
            write_clock()
CodexPlanner.solve=solve
original=ToolResult.__post_init__
out=Path(os.environ['RPENT_RUN_OUTPUT'])
def observed(self):
    original(self)
    for block in self.content_blocks:
        if block.get('type') != 'text':
            continue
        try:
            value=json.loads(block.get('text',''))
        except (ValueError,TypeError):
            continue
        if not isinstance(value,dict):
            continue
        if value.get('terminated') is True and 'state' in value:
            mark_end('environment_terminated')
        elif value.get('truncated') is True and 'state' in value:
            mark_end('environment_truncated')
        elif value.get('_finish') is True:
            mark_end('finish:'+str(value.get('status')))
    blocks=[]
    for b in self.content_blocks:
        if b.get('type')=='image':
            data=base64.b64decode(b['source']['data']);blocks.append(dict(type='image',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        else:blocks.append(b)
    with (out/'tool-results.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),name=self.name,blocks=blocks),ensure_ascii=False)+'\n')
ToolResult.__post_init__=observed
from rpent.cli.main import main
raise SystemExit(main())
