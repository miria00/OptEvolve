"""Write noisy test images as .npy so the solver env needs no imaging stack."""
import argparse, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="")
ap.add_argument("--outdir", default="/home/miria/data/tv")
ap.add_argument("--sizes", default="128,256,512,1024,2048")
ap.add_argument("--sigma", type=float, default=0.1)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
if a.src:
    import imageio.v3 as iio
    im = iio.imread(a.src).astype(np.float64)
    if im.ndim == 3: im = im.mean(axis=2)
    im /= 255.0
    tag = os.path.splitext(os.path.basename(a.src))[0]
else:
    from skimage import data
    im = data.retina().astype(np.float64).mean(axis=2) / 255.0
    tag = "retina"
rng = np.random.default_rng(0)
for s in [int(v) for v in a.sizes.split(",")]:
    if im.shape[0] < s or im.shape[1] < s:
        print("skip %d (source %s)" % (s, im.shape)); continue
    c = im[:s, :s].copy()
    y = c + a.sigma * rng.standard_normal(c.shape)
    np.save(os.path.join(a.outdir, f"{tag}_{s}_clean.npy"), c)
    np.save(os.path.join(a.outdir, f"{tag}_{s}_noisy.npy"), y)
    print("wrote %s %d x %d" % (tag, s, s))
