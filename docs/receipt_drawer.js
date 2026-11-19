'use strict';
(() => {
  const caption = document.querySelector('#figB figcaption');
  const button = document.createElement('button');
  button.id = 'viewReceipt'; button.textContent = 'View receipt'; button.type = 'button';
  button.style.marginLeft = 'auto'; caption.append(button);
  const drawer = document.createElement('dialog'); drawer.id = 'receiptDrawer';
  drawer.style.cssText = 'position:fixed;inset:40px 24px 40px auto;margin:0;width:660px;max-width:90vw;padding:30px;border:1px solid var(--rule);background:var(--paper);color:var(--ink);box-shadow:0 24px 80px #1B365D28;font:15px/1.5 var(--sans)';
  const title = document.createElement('h2');title.textContent='Verified receipt';title.style.cssText='font:38px var(--serif);margin:0 0 20px';
  const close = document.createElement('button');close.textContent='Close';close.style.float='right';close.onclick=()=>drawer.close();
  const fields = document.createElement('div');fields.id='receiptFields';
  drawer.append(close,title,fields);document.body.append(drawer);
  button.onclick = () => {
    const r=window.RECORDED_RUNS[3],v=window.workbenchRecompute(r);
    fields.replaceChildren();
    const data={scenario:'4 · Fake green caught',hardware:v.hardware_path,workspace_tests:'PASS',frozen_tests:'FAIL',sqlite_total:r.checks.sqlite.rows[0][3]+' ≠ 42',protected_unchanged:String(r.checks.filesystem.protected_unchanged),model_sha256:r.model.sha256,policy_sha256:r.boundary.policy_sha256,protected_sha256:r.checks.filesystem.protected_sha256,prompt_sha256:r.context.prompt_sha256};
    Object.entries(data).forEach(([k,val])=>{const row=document.createElement('div');row.style.cssText='padding:12px 0;border-top:1px solid var(--hair)';const name=document.createElement('div');name.textContent=k;name.style.cssText='font:12px var(--mono);color:var(--quiet)';const value=document.createElement('div');value.textContent=val;value.style.cssText='font:16px var(--mono);overflow-wrap:anywhere';row.append(name,value);fields.append(row)});
    drawer.showModal();
  };
})();
