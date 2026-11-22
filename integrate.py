"""Night integration: fold every finished RS model into the ensemble and pick the better of two variants per method by
the full-pipeline CV (pipecv.py: gate -> box -> zoom -> SAM -> map, all thresholds fit on training folds).
    python integrate.py [methods...]
Per method:
  full  = make_ensrs(old ensemble, ALL clean RS full maps e2eRS[z|f]<tag>_<m>, old weight 1/3, gate-full images)
  part  = make_ensrs(full, ALL RS partial maps e2eRP[z|f]<tag>_<m>, old weight 1/3, gate-partial images)
Both tags are copied to results/preds; pipecv is run on each with the method's partial settings; the best tag is
printed with its CV.  Maps from the buggy multi-variant scripts (realsim_*, realsynth_post/pre_*, e2eRSpost) are ignored.
Writes results/integrate_<method>.txt: best tag, CV, spec for make_sub3.py."""
import glob, os, re, shutil, subprocess, sys
import data

OLD = {"pixelseal": "ens3_pixelseal", "trustmark": "ens2_trustmark", "stegastamp": "ens2_stegastamp"}
SPEC = {"pixelseal": (os.environ.get("PS_BOX", "brute"), ""), "trustmark": ("brute", "e2eR512s1Z_trustmark"),
        "stegastamp": ("rsbox", "e2eR512s1Z_stegastamp")}
OUT = "research/out"


def tags(prefix, m):
    fs = sorted(glob.glob(f"{OUT}/{prefix}*_{m}_test.npz"))
    t = [os.path.basename(f)[:-len("_test.npz")] for f in fs]
    return [x for x in t if os.path.exists(f"{OUT}/{x}_train.npz") and "post" not in x and "pre" not in x.split("_")[0][5:] and not re.match(r"e2eR[SP][zf]L", x)]   # L* = low-res (weaker)


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


MAPGATE = {"trustmark", "stegastamp"}          # map-mean gate measured better for these two (06:50: +0.006 / +0.005)


def cv(m, tag):
    box, zoom = SPEC[m]; g = f"GATE_MAP={m}={tag} " if m in MAPGATE else ""
    out = run(f"{g}python pipecv.py --method {m} --map {tag} --box '{box}' --zoom '{zoom}' --sam 1 2>&1")
    v = re.search(r"CV ([0-9.]+)", out)
    print("  ", out.strip().splitlines()[0] if out.strip() else out, flush=True)
    return float(v.group(1)) if v else -1


for m in (sys.argv[1:] or data.METHODS):
    full_rs = [t for t in tags("e2eRS", m) if re.match(r"e2eRS[zf]", t)]
    part_rs = [t for t in tags("e2eRP", m) if re.match(r"e2eRP[zf]", t)]
    print(f"== {m}: {len(full_rs)} RS full maps {full_rs}\n   {len(part_rs)} RS partial maps {part_rs}", flush=True)
    # fixed candidate set (decided before seeing the night's new models): N = old 1/3 + all RS full maps;
    # W = old 0.2 + fine-tuned RS full maps only; *P = plus all fine-tuned RS-partial maps on gate-partial images.
    fine = [t for t in full_rs if t.startswith("e2eRSf")]; pfine = [t for t in part_rs if t.startswith("e2eRPf")]
    cands = []
    for name, maps, w in (("ensN", full_rs, 0.333), ("ensW", fine, 0.2)):
        tag = f"{name}_{m}"; run(f"python research/tests/make_ensrs.py {m} {OLD[m]} {','.join(maps)} {tag} {w} full"); cands.append(tag)
        if pfine:
            ptag = f"{name}P_{m}"; run(f"python research/tests/make_ensrs.py {m} {tag} {','.join(pfine)} {ptag} 0.333 partial"); cands.append(ptag)
    for t in cands:
        for s in ("train", "test"):
            dst = data.RESULTS / "preds" / f"{t}_{s}.npz"
            if not dst.exists():
                os.symlink(f"../../{OUT}/{t}_{s}.npz", dst)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        res = dict(zip(cands, ex.map(lambda t: cv(m, t), cands)))
    best = max(res, key=res.get); box, zoom = SPEC[m]
    spec = best + (f":{box}" if box else "") + (f":zoom={zoom}" if zoom else "") + ":sam"
    open(data.RESULTS / f"integrate_{m}.txt", "w").write(f"{best} {res[best]:.4f} {spec} {'GATE_MAP=' + m + '=' + best if m in MAPGATE else ''}\n")
    print(f"   BEST {m}: {best} CV {res[best]:.4f}  spec {spec}", flush=True)
