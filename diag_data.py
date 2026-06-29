import numpy as np
from core.head_direction_preprocessor import HeadDirectionPreprocessor

p = HeadDirectionPreprocessor(session_path="th1_data/Mouse12-120806",
                              bin_size=0.025, smoothing_std=0.05,
                              target_structure="adn", cache_dir="th1_data")
spks, mov, params = p.generate_dataset()

ang = np.deg2rad(params["orientation"].to_numpy().astype(np.float64))
c, s = np.cos(ang), np.sin(ang)
print("\n== samples", spks.shape, "angle std %.2f ==" % ang.std())

z = (spks - spks.mean(0, keepdims=True)) / (1e-9 + spks.std(0, keepdims=True))
z20 = z[:, :20]

def bestcorr(M, t):
    t = t - t.mean()
    num = np.abs(M.T @ t); den = np.linalg.norm(M, axis=0)*np.linalg.norm(t)+1e-9
    return float(np.nanmax(num/den)), int(np.nanargmax(num/den))

bc, ic = bestcorr(z20, c); bs, isn = bestcorr(z20, s)
print("best |corr| neuron~cos: %.4f (neuron %d)" % (bc, ic))
print("best |corr| neuron~sin: %.4f (neuron %d)" % (bs, isn))

X = np.c_[z20, np.ones(len(z20))]
wc = np.linalg.lstsq(X, c, rcond=None)[0]
ws = np.linalg.lstsq(X, s, rcond=None)[0]
pred = np.arctan2(X@ws, X@wc)
def circ_mean(a): return np.angle(np.mean(np.exp(1j*a)))
da = np.sin(ang-circ_mean(ang)); db = np.sin(pred-circ_mean(pred))
rcc = np.sum(da*db)/np.sqrt(np.sum(da**2)*np.sum(db**2)+1e-12)
print("linear population decode r_cc: %.4f" % abs(rcc))
