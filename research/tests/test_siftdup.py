"""Kill test (written before running): do images share source content at a different scale / position
(outpainted rectangle = shrunk copy of another dataset image; transplanted object from another dataset image)?
SIFT (cv2), 400 kp/img, mutual NN ratio test 0.75 over all 1530 images, RANSAC similarity on candidate pairs.
PASS if >= 5 % of images have a partner with >= 25 RANSAC inliers AND (train) the inlier hull in the partial image
overlaps the GT watermark mask (>= 70 % of inliers inside the mask) in most matched train pairs. Else KILL."""
import sys, numpy as np, cv2, torch
cv2.setNumThreads(1)
sys.path.insert(0, "benelhajyahia1"); import data
from multiprocessing import Pool
def feats(args):
    sp, i = args
    ds = data.load_split(sp); it = ds[i]
    g = cv2.cvtColor(it["image"], cv2.COLOR_RGB2GRAY)
    s = cv2.SIFT_create(nfeatures=400); kp, d = s.detectAndCompute(g, None)
    if d is None: d = np.zeros((0, 128), np.float32); kp = []
    d = d / (np.abs(d).sum(1, keepdims=True) + 1e-7); d = np.sqrt(d)  # RootSIFT
    m = it.get("mask")
    return it["image_name"], sp, np.array([k.pt for k in kp], np.float32), d.astype(np.float32), m
if __name__ == "__main__":
    jobs = [("train", i) for i in range(510)] + [("test", i) for i in range(1020)]
    with Pool(16) as p: R = p.map(feats, jobs, chunksize=8)
    owner = np.concatenate([np.full(len(r[3]), k) for k, r in enumerate(R)])
    D = torch.from_numpy(np.concatenate([r[3] for r in R])); P = np.concatenate([r[2] for r in R])
    N = len(D); print("descriptors", N, flush=True)
    own = torch.from_numpy(owner)
    nn1 = np.zeros(N, int); ok = np.zeros(N, bool)
    torch.set_num_threads(16)
    for s in range(0, N, 4096):
        q = D[s:s+4096]; dist = torch.cdist(q, D)
        dist[own[s:s+4096, None] == own[None, :]] = 1e9          # exclude same image
        v, ix = dist.topk(2, largest=False)
        nn1[s:s+4096] = ix[:, 0].numpy(); ok[s:s+4096] = (v[:, 0] < 0.75 * v[:, 1]).numpy()
    pairs = {}
    for a in np.where(ok)[0]:
        key = (owner[a], owner[nn1[a]]); pairs.setdefault(key, []).append((a, nn1[a]))
    res = []
    for (i, j), lst in pairs.items():
        if len(lst) < 8: continue
        A = P[[a for a, b in lst]]; B = P[[b for a, b in lst]]
        M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=4)
        if M is None: continue
        n = int(inl.sum()); sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        insideA = None
        if R[i][4] is not None and n:
            pa = A[inl[:, 0] > 0].astype(int).clip(0, 511); insideA = float(R[i][4][pa[:, 1], pa[:, 0]].mean())
        res.append((n, R[i][0], R[i][1], R[j][0], R[j][1], sc, insideA))
    res.sort(reverse=True)
    good = [r for r in res if r[0] >= 25]
    imgs = set(r[1] for r in good) | set(r[3] for r in good)
    print("pairs with >=8 ratio matches", len(res), " >=25 inliers", len(good), " images involved", len(imgs), "/1530")
    for r in res[:40]: print(r)
    np.save("benelhajyahia1/research/out/siftdup.npy", np.array(res, dtype=object), allow_pickle=True)
