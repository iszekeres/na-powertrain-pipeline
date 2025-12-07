#!/usr/bin/env python3
import argparse, os, subprocess, json, sys, time, shutil, re
from collections import deque

FRAMES = "|/-\\"

def bar(done, total, width=28):
    if total <= 0: total = 1
    done = max(0, min(done, total))
    pct = float(done)/float(total)
    fill = int(width*pct)
    return "█"*fill + "·"*(width-fill), int(pct*100)

def run_with_subprogress(label, cmd, step_idx, step_total):
    # launch with live stdout
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         bufsize=1, universal_newlines=True)

    t0 = time.perf_counter()
    spin = 0
    sub_cur = 0
    sub_tot = 0
    last_lines = deque(maxlen=12)
    term_cols = shutil.get_terminal_size((120, 24)).columns

    def render(final=False, rc=0):
        outer, _ = bar(step_idx - (0 if final else 1), step_total, width=28)
        if sub_tot > 0:
            inner, _ = bar(sub_cur, sub_tot, width=24)
            status = f"[{outer}] {step_idx}/{step_total} {label}  {FRAMES[spin%len(FRAMES)] if not final else ('✓' if rc==0 else f'✗ ({rc})')}  | [{inner}] {sub_cur}/{sub_tot}"
        else:
            status = f"[{outer}] {step_idx}/{step_total} {label}  {FRAMES[spin%len(FRAMES)] if not final else ('✓' if rc==0 else f'✗ ({rc})')}"
        if not final:
            print("\r"+status[:term_cols-1], end="", flush=True)
        else:
            print("\r"+" "*(term_cols-1), end="\r")
            print(status[:term_cols-1], flush=True)

    # stream reader loop
    while True:
        line = p.stdout.readline()
        if line == "" and p.poll() is not None:
            break
        if line:
            s = line.rstrip("\r\n")
            last_lines.append(s)
            # detect "(x/n)" anywhere in the line
            m = re.search(r"\((\d+)\s*/\s*(\d+)\)", s)
            if m:
                try:
                    sub_cur = int(m.group(1))
                    sub_tot = max(1, int(m.group(2)))
                except Exception:
                    pass
        render(final=False)
        time.sleep(0.08); spin += 1

    rc = p.wait()
    render(final=True, rc=rc)

    if rc != 0:
        print("---- last output ----", file=sys.stderr)
        for ln in last_lines:
            print(ln, file=sys.stderr)
        print("---------------------", file=sys.stderr)
    return rc

def maybe_weighted(script):
    base = script[:-3] if script.endswith('.py') else script
    cand = base + '_weighted.py'
    return cand if os.path.exists(cand) else script

def main():
    ap=argparse.ArgumentParser(description="Weighted tune pass builder (profiles-driven) with subprogress.")
    ap.add_argument('--profile', required=True, help='Profile from tune_pass_profiles.json (e.g., comfort_ultraplush, comfort, tow, performance, comfort_lockouts)')
    ap.add_argument('--logs-glob', default=r'.\06_Logs\Trans_Review\__trans_focus__clean__*.csv')
    ap.add_argument('--dir', default=r'.\06_Logs\Trans_Review')
    ap.add_argument('--profiles-json', default=r'.\tune_pass_profiles.json')
    ap.add_argument('--half-life-days', type=float, default=30.0)
    ap.add_argument('--route-bias', default='neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1')
    ap.add_argument('--cap-up', type=float, default=0.6)
    ap.add_argument('--cap-down', type=float, default=0.6)
    ap.add_argument('--cap-tcc', type=float, default=0.8)
    args=ap.parse_args()

    if not os.path.exists(args.profiles_json):
        print(f"[ERROR] profiles JSON not found: {args.profiles_json}", file=sys.stderr); sys.exit(2)

    with open(args.profiles_json,'r',encoding='utf-8') as f:
        profs=json.load(f)

    if args.profile not in profs:
        print(f"[ERROR] profile '{args.profile}' not found. Available: {', '.join(sorted(profs.keys()))}", file=sys.stderr); sys.exit(2)

    P = profs[args.profile]
    passes = P.get('passes', [])

    # Build the step list (all passes + blend + polish + audit/pack)
    steps = []
    for item in passes:
        script = maybe_weighted(item['script'])
        extra = item.get('args', {})
        outp = '.\\\\' + os.path.basename(script).split('.')[0].upper().replace('_PASS','').replace('_HELPER','').replace('_DOWNHILL','')
        cmd = f'python .\\\\{script} --logs-glob "{args.logs_glob}" --out-prefix "{outp}" --half-life-days {args.half_life_days} --route-bias "{args.route_bias}"'
        for k,v in extra.items():
            cmd += f" {k} {v}"
        steps.append((f"RUN {os.path.basename(script)}", cmd))

    steps.append(("BLEND deltas", f'python .\\\\delta_blend.py --dir "{args.dir}" --nvh-prefix ".\\\\NVH" --therm-prefix ".\\\\THERM" --cap-up {args.cap_up} --cap-down {args.cap_down} --cap-tcc {args.cap_tcc}'))
    steps.append(("POLISH overlay", f'python .\\\\overlay_polish_v3.py --dir "{args.dir}"'))
    lock_flag = '--tcc-comfort-lockouts' if P.get('tcc_lockouts', False) else ''
    steps.append(("AUDIT & PACK", f'python .\\\\audit_and_pack.py --dir "{args.dir}" {lock_flag} --down-gap-floor 1.6 --tcc-release-gap-floor 1.2'))

    total = len(steps)
    print(f"[INFO] Using profile: {args.profile} | Steps: {total}", flush=True)

    # Run steps with subprogress
    for i,(label, cmd) in enumerate(steps, start=1):
        rc = run_with_subprogress(label, cmd, i, total)
        if rc != 0:
            print(f"[STOP] Halting on failure of: {label}", file=sys.stderr)
            sys.exit(rc)

    b, _ = bar(total, total)
    print(f"[{b}] {total}/{total} DONE  ✓  All steps complete.", flush=True)

if __name__=='__main__':
    main()
