
import sys, json, numpy as np, torch, os, tempfile
from third_party import MoGe   

if len(sys.argv) != 2:
    print("usage: python moge_worker.py in.npy", file=sys.stderr); sys.exit(1)

in_path = sys.argv[1]
print("[worker] loading", in_path, file=sys.stderr, flush=True)

x = np.load(in_path)                 # (B,C,H,W) or (C,H,W)
print("[worker] shape", x.shape, file=sys.stderr, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MoGe(cache_dir="/tmp/moge_cache").to(device).eval()

with torch.no_grad():
    pred_p, pred_m = model.forward_image(torch.from_numpy(x).to(device))
                                  
out_p = tempfile.NamedTemporaryFile(delete=False, suffix=".npy").name
out_m = tempfile.NamedTemporaryFile(delete=False, suffix=".npy").name
np.save(out_p, pred_p.cpu().numpy())
np.save(out_m, pred_m.cpu().numpy())

print(json.dumps({"p": out_p, "m": out_m}), flush=True) 
