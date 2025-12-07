
#!/usr/bin/env python3
# tune_pass_builder_weighted.py — same as builder, with weighting flags & caps passed through.
import argparse, os, subprocess, json

def run(cmd):
    print("[RUN]", cmd, flush=True)
    ec = subprocess.call(cmd, shell=True)
    if ec!=0: print(f"[WARN] Step returned {ec}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["comfort","performance","tow"])
    ap.add_argument("--logs-glob", default=r".\06_Logs\Trans_Review\__trans_focus__clean__*.csv")
    ap.add_argument("--dir", default=r".\06_Logs\Trans_Review")
    ap.add_argument("--profiles-json", default=r".\tune_pass_profiles.json")
    ap.add_argument("--half-life-days", type=float, default=30.0)
    ap.add_argument("--route-bias", default="neighborhood=1.5,inbound=1.2,outbound=1.2,highway=1.1")
    ap.add_argument("--cap-up", type=float, default=0.6)
    ap.add_argument("--cap-down", type=float, default=0.6)
    ap.add_argument("--cap-tcc", type=float, default=0.8)
    args = ap.parse_args()

    with open(args.profiles_json,"r",encoding="utf-8") as f:
        profs = json.load(f)
    P = profs[args.profile]

    # Run passes (swap to weighted NVH/THERM if present)
    for item in P["passes"]:
        script = item["script"]
        if script=="nvh_pass.py" and os.path.exists(".\\nvh_pass_weighted.py"):
            script="nvh_pass_weighted.py"
        if script=="thermal_watch.py" and os.path.exists(".\\thermal_watch_weighted.py"):
            script="thermal_watch_weighted.py"
        extra = item.get("args", {})
        outp = ".\\" + script.split(".")[0].upper().replace("_PASS","").replace("_HELPER","").replace("_DOWNHILL","")
        cmd = f'python .\\{script} --logs-glob "{args.logs_glob}" --out-prefix "{outp}"'
        # add weighting flags if script supports them
        if "nvh_pass" in script or "thermal_watch" in script or "corner_exit" in script or "rpm_floor_guard" in script or "stopngo" in script or "driver_intent" in script or "traction" in script or "engine_brake" in script or "dfco" in script or "shift_consistency" in script or "shift_latency" in script or "occupancy_weight" in script:
            cmd += f" --half-life-days {args.half_life_days} --route-bias \"{args.route_bias}\""
        for k,v in extra.items():
            cmd += f" {k} {v}"
        run(cmd)

    # Blend with caps
    run(f'python .\\delta_blend.py --dir "{args.dir}" --nvh-prefix ".\\NVH" --therm-prefix ".\\THERM" --cap-up {args.cap_up} --cap-down {args.cap_down} --cap-tcc {args.cap_tcc}')

    # Re-polish & Audit+Pack
    run(f'python .\\overlay_polish_v3.py --dir "{args.dir}"')
    lock_flag = "--tcc-comfort-lockouts" if P.get("tcc_lockouts", False) else ""
    run(f'python .\\audit_and_pack.py --dir "{args.dir}" {lock_flag} --down-gap-floor 1.6 --tcc-release-gap-floor 1.2')
    print(f"[DONE] Weighted tune build complete for profile: {args.profile}")

if __name__=="__main__":
    main()
