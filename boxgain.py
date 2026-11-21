"""Exit 0 if the boxes in results/<BRUTE_TAG>_<method>_train.npz beat the current boxes (results/brute_*) in CV by > 0.005."""
import os, sys
import recteval
method, tag = sys.argv[1], sys.argv[2]
os.environ["BRUTE_TAG"] = "brute"; recteval.main(tag, "score"); old = recteval.main.result[1]
os.environ["BRUTE_TAG"] = sys.argv[3]; recteval.main(tag, "score"); new = recteval.main.result[1]
print(f"BOXGAIN {method}: old boxes {old:.4f} -> new boxes {new:.4f}", flush=True)
sys.exit(0 if new > old + 0.005 else 1)
