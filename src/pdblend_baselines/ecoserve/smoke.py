"""Small native-v1 EcoServe request runner; no profile means no run."""
import argparse, asyncio, json, os, time
from .runtime import HttpEcoServeTransport, EcoServeCapabilityError
from .policy import PrefillProfile

async def run(args):
    if not args.profile or not os.path.isfile(args.profile):
        raise SystemExit(f"missing_profile: {args.profile or '<unspecified>'}")
    try: profile=PrefillProfile.load(args.profile)
    except Exception as exc: raise SystemExit(f"missing_profile: {args.profile}: {exc}")
    transport=HttpEcoServeTransport(args.url); events=[]
    payload={"request_id":args.request_id,"model":args.model,
             "prompt":list(range(args.prompt_tokens)),"max_tokens":args.max_tokens,
             "system":"ecoserve","profile":os.path.abspath(args.profile),"tp":args.tp}
    async for event in transport.generate(payload):
        events.append(event)
        if event.get("finished") or event.get("finish_reason"): break
    if not events or not any(e.get("finished") or e.get("finish_reason") for e in events):
        raise EcoServeCapabilityError("native /baseline/generate ended without terminal token event")
    result={"system":"ecoserve","model":args.model,"tp":args.tp,
            "profile":os.path.abspath(args.profile),"profile_sha256":profile.source_sha256,
            "request_id":args.request_id,"events":events,"completed":True,
            "mechanism_validated":False,"qualification":"ordinary_generate_only","at_s":time.time()}
    with open(args.out,"w") as f: json.dump(result,f,indent=2)
    return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--url",required=True);p.add_argument("--model",required=True)
    p.add_argument("--profile");p.add_argument("--tp",type=int,default=1);p.add_argument("--prompt-tokens",type=int,default=16)
    p.add_argument("--max-tokens",type=int,default=16);p.add_argument("--request-id",default="ecoserve-smoke");p.add_argument("--out",required=True)
    print(json.dumps(asyncio.run(run(p.parse_args(argv))),sort_keys=True))
if __name__=="__main__":main()
