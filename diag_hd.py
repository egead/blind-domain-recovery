import os, json, numpy as np, tensorflow as tf
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from core.train_utils import create_model, make_data_generator
from core.circular_evaluation import recover_phase, best_circular_correlation

EXP = "experiments/HEADDIRECTION-EXP-0"
specs = json.load(open(os.path.join(EXP, "specs.json")))
seed = specs.get("seed", 0)
np.random.seed(seed); tf.random.set_seed(seed)

model = create_model(**specs["model_params"])
dg0 = make_data_generator(seed=seed, **specs["data_generator_params"])
_ = model(tf.constant(dg0.sample_batch_of_data(), tf.float32), training=False)
model.load_weights(os.path.join(EXP, "epochs/ep300.h5"))
print("weights loaded")

dg = make_data_generator(seed=seed + 123, **specs["data_generator_params"])
dg.reset_batch_counter()
x_hd, params_hd = dg.sample_batch_of_data(return_hidden_params=True)
true_angle = np.deg2rad(params_hd[:, 0].astype(np.float64))
c, s = np.cos(true_angle), np.sin(true_angle)

print("\n== INPUT shapes ==")
print("x_hd", x_hd.shape, "true_angle", true_angle.shape,
      "angle spread min %.2f max %.2f std %.2f" % (true_angle.min(), true_angle.max(), true_angle.std()))

def best_corr(M, target):
    M = M - M.mean(0, keepdims=True)
    t = target - target.mean()
    num = np.abs(M.T @ t)
    den = np.linalg.norm(M, axis=0) * np.linalg.norm(t) + 1e-9
    return np.nanmax(num / den)

print("\n== RAW NEURONS vs head direction ==")
print("best |corr| neuron~cos:", round(best_corr(x_hd, c), 4))
print("best |corr| neuron~sin:", round(best_corr(x_hd, s), 4))

y_plain = model(tf.constant(x_hd, tf.float32), training=False).numpy()
y_an, L = model(tf.constant(x_hd, tf.float32), training=False, analyze=True)
y_an = y_an.numpy()
print("\n== LIFTED shapes ==")
print("plain call :", y_plain.shape)
print("analyze    :", y_an.shape, " L:", L.numpy().shape)

flat = y_an.reshape(y_an.shape[0], -1)
print("\n== LIFTED features vs head direction ==")
print("best |corr| lifted~cos:", round(best_corr(flat, c), 4))
print("best |corr| lifted~sin:", round(best_corr(flat, s), 4))

print("\n== recover_phase r_cc under different axis choices ==")
def rcc(rec): return round(best_circular_correlation(true_angle, rec)[0], 4)
print("analyze, channel=0           :", rcc(recover_phase(y_an, channel=0)))
yp2 = y_plain[:, 0, :] if y_plain.ndim == 3 else y_plain
print("plain squeezed (b,positions) :", rcc(recover_phase(yp2)))
print("plain, channel=0 (orig call) :", rcc(recover_phase(y_plain, channel=0)))

prof = y_an[:, :, 0]
prof = prof - prof.mean(1, keepdims=True)
peak = np.argmax(prof, 1).astype(float)
peak_phase = 2 * np.pi * peak / prof.shape[1]
print("\n== bump-peak position vs angle ==")
print("peak-position r_cc           :", rcc(peak_phase))
print("profile per-sample std (mean):", round(float(np.std(prof, 1).mean()), 4))
