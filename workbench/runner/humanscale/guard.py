"""Apply the same boundary to an explicit external agent or MCP command."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from .observer import process_tree, PACKAGE
from .sandbox import profile
from .vendor.gpu_meters import _IOAccel


def guard_command(command, workspace, model, port, output, timeout=150):
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('Supply an explicit agent command after --')
    executable=Path(shutil.which(command[0]) or command[0]).resolve()
    if not executable.is_file():raise ValueError('Executable not found')
    workspace=Path(workspace).resolve();workspace.mkdir(parents=True,exist_ok=True)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    if output.is_relative_to(workspace):raise ValueError('Observer output must be outside the writable workspace')
    policy=profile(workspace,model,PACKAGE,port)
    policy+='(allow process-exec file-read* (literal '+json.dumps(str(executable))+'))\n'
    (output/'policy.sb').write_text(policy)
    env={**os.environ,'HOME':str(workspace),'TMPDIR':str(workspace),'PYTHONDONTWRITEBYTECODE':'1'}
    p=subprocess.Popen(['/usr/bin/sandbox-exec','-f',str(output/'policy.sb'),str(executable),*command[1:]],cwd=workspace,env=env,start_new_session=True)
    started=time.time();trees=[];samples=[];reason=None
    try:
        try:meter=_IOAccel()
        except OSError:meter=None
        while p.poll() is None:
            tree=process_tree(p.pid);trees.append({'at':time.time(),'processes':tree})
            sample=meter.sample() if meter else None
            if sample:samples.append({**sample,'at':time.time()})
            if any(x['rss_bytes']>4*1024**3 for x in tree):reason='memory_budget'
            if time.time()-started>timeout:reason='timeout'
            if reason:
                os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(.15)
        p.wait()
    finally:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    receipt={'schema':'humanscale.guard.v1','returncode':p.returncode,'command_exit_code':p.returncode,
             'command_completed':p.returncode==0 and reason is None,'boundary_enforced':True,'termination':reason or 'complete',
             'memory_budget_bytes':4*1024**3,'process_tree':trees,'os_samples':samples,
             'os_sampling_state':'OBSERVED' if samples else 'HOST_VERIFIED',
             'independent_task_outcome':'VERIFIED' if p.returncode==0 and reason is None else 'COMMAND_NONZERO',
             'offloaded_layers':None,
             'note':'Kernel Seatbelt boundary enforced around child command process tree.'}
    (output/'guard.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
