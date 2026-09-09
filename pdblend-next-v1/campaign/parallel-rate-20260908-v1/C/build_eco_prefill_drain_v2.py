"""Keep admitted prefill runnable while EcoServe changes the selected window."""
import ast,hashlib,json,shutil,time
from pathlib import Path
C=Path(__file__).resolve().parent;REPO=C.parents[2]
OLD=REPO/'releases/five-system100-C7B-baseline-v1-runtime'
NEW=REPO/'releases/five-system100-C7B-baseline-eco-drain-v2-runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((OLD/'manifest.json').read_text());assert not NEW.exists();NEW.mkdir()
for name,h in old['files'].items():
 src=OLD/name;assert sha(src)==h;dest=NEW/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
runtime=NEW/'src/ecopadg/serving/runtime.py';before=runtime.read_text()
helper='''    async def eco_require_prefill_drained(self, plan):
        """Closing a window must not strand its already admitted prefills.

        EcoServe's instance scheduler prioritizes accepted prefills (§3.3).
        The shared temporal engine's OFF switch pauses native waiting work,
        so defer this switch before any reservation or physical command until
        that admitted prefill batch has produced its first tokens. Decode work
        already in progress may continue; no latency/deadline is extended.
        Caller owns action_lock. Native and stream telemetry remain runnable.
        """
        closing={a.instance_id for a in plan.windows if not a.admit_prefill}
        if not closing:
            return
        instances={i.instance_id:i for i in self.state.snapshot.instances}
        protected=[]
        for identifier in closing:
            instance=instances.get(identifier)
            if instance is None or not 0<=time.time()-instance.timestamp_s<=1:
                protected.append(identifier)
            elif instance.admit_prefill and (instance.waiting>0 or
                    any(r.first_token_s is None for r in instance.requests)):
                protected.append(identifier)
        if protected:
            self.planning_stats.counts['eco_window_close_deferred']+=1
            # Yield to native completions/first-token publication before the
            # original ExpiredPlan path requeues the unchanged request.
            await asyncio.sleep(0)
            raise ExpiredPlan('EcoServe admitted prefill must drain before window closure: '+','.join(sorted(protected)))

'''
anchor='    async def eco_resize(self):\n';assert before.count(anchor)==1
text=before.replace(anchor,helper+anchor)
anchor='                            admitted=admission_budget(plan,request,now=committed_s)\n';assert text.count(anchor)==1
text=text.replace(anchor,"                            if self.eco_scheduler:\n                                await self.eco_require_prefill_drained(plan)\n"+anchor)
anchor='''                        try:
                            await self.backend.execute(plan)
                        except ExpiredPlan:
                            # A pre-action expiry changed no engine window.''';assert text.count(anchor)==1
text=text.replace(anchor,'''                        try:
                            await self.eco_require_prefill_drained(plan)
                            await self.backend.execute(plan)
                        except ExpiredPlan:
                            # A pre-action expiry changed no engine window.''')
ast.parse(text);runtime.write_text(text)
files={name:sha(NEW/name) for name in old['files']};changed=[name for name in files if files[name]!=old['files'][name]];assert changed==['src/ecopadg/serving/runtime.py']
manifest={**old,'files':files,'parent_release':str(OLD),'parent_manifest_sha256':sha(OLD/'manifest.json'),'candidate_release':str(NEW),'change':'EcoServe defer unsafe temporal OFF before any reservation/physical action; admission and membership use same guard','created_s':time.time(),'declared_strategy_scope':'ecoserve only','paper_reference':'https://www.usenix.org/system/files/osdi26-du.pdf#page=7','source_builder':dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))}
(NEW/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(host=str(NEW),manifest_sha256=sha(NEW/'manifest.json'),runtime_sha256=sha(runtime),changed=changed)))
