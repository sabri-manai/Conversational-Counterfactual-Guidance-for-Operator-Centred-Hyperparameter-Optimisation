"""
patchcore_review_lib.py
=======================
Single, tested implementation of the PatchCore pipeline used by the review notebooks
(R01-R03).  It is a faithful copy of the building blocks in

    mvtec/01_data_collection.ipynb      (HPO objective, per-category protocol)
    mvtec/05_confirmation_evaluation.ipynb  (pooled confirmation protocol)
    proditec/01_data_collection.ipynb   (HPO objective)
    proditec/05_confirmation_evaluation.ipynb

with three deliberate additions:
  * ``run_seed``      - one knob that re-seeds the train sub-sample, the patch sub-sample
                        and the coreset draw (default SEED=42 reproduces the original code).
  * ``remap_*``       - the split CSVs store absolute paths of the original server; these
                        helpers re-root them onto a local copy of the dataset.
  * ``run_per_category`` - the protocol that matches the MVTec HPO objective (one PatchCore
                        and one review-budget threshold per category, mean of the F1s).

Nothing here changes the modelling choices of the paper.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYERS_MAP = {
    "l2_l3": ("layer2", "layer3"),
    "l1_l2_l3": ("layer1", "layer2", "layer3"),
    "l3": ("layer3",),
    "l2": ("layer2",),
}
IMAGE_SIZE_MAP = {"224": (224, 224), "256": (256, 256), "320": (320, 320)}
MVTEC_CLASSES = ["bottle", "cable", "capsule", "hazelnut"]

# A "reasonable default" PatchCore, fixed a priori (NOT tuned on any data here):
# WideResNet50 layers 2+3, 256 px resized + 0.875 centre-crop (= 224), 10 % coreset (the upper
# end of the searched range), 1 nearest neighbour, max reduction, all training data, no corruption.
# The review budget is NOT a model hyper-parameter; it is supplied by the caller.
REFERENCE_CFG = {
    "backbone": "wide_resnet50_2", "layers_key": "l2_l3", "pre_trained": True,
    "image_size_key": "256", "center_crop_key": "0.875",
    "coreset_sampling_ratio": 0.101, "num_neighbors": 1, "reduction": "max",
    "max_patches_per_image": 1024, "batch_size": 16,
    "soft_train_fraction": 1.0, "soft_corruption_level": "none",
}


def set_device(name: str | None) -> torch.device:
    global DEVICE
    DEVICE = torch.device(name) if name else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
def remap_path(p: str, marker: str, root: str | Path | None) -> str:
    """Re-root an absolute path of the original server onto a local dataset copy."""
    p = str(p)
    if root is None or Path(p).exists():
        return p
    i = p.find(marker)
    if i < 0:
        return p
    return str(Path(root) / p[i + len(marker):].lstrip("/"))


def _read_split(csv_path: Path, marker: str, root) -> pd.DataFrame:
    d = pd.read_csv(csv_path)
    d["label"] = pd.to_numeric(d["label"], errors="coerce").astype(int)
    d["path"] = d["path"].map(lambda x: remap_path(x, marker, root))
    return d


def load_mvtec_class_data(splits_dir: Path, mvtec_root=None, classes=MVTEC_CLASSES) -> dict:
    """{class: {"train": df, "val": df, "final": df}} with paths re-rooted onto ``mvtec_root``."""
    out = {}
    for cls in classes:
        d = Path(splits_dir) / cls
        out[cls] = {
            "train": _read_split(d / "df_train_good.csv", "mvtec_anomaly_detection/", mvtec_root),
            "val": _read_split(d / "df_val.csv", "mvtec_anomaly_detection/", mvtec_root),
            "final": _read_split(d / "df_final.csv", "mvtec_anomaly_detection/", mvtec_root),
        }
    return out


def pool_class_data(class_data: dict) -> dict:
    """Concatenate categories exactly as mvtec/05_confirmation_evaluation.ipynb does."""
    parts = {k: [] for k in ("train", "val", "final")}
    for cls, d in class_data.items():
        for k in parts:
            x = d[k].copy()
            x["class_name"] = cls
            parts[k].append(x)
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def load_proditec_data(splits_dir: Path, pdt_root=None) -> dict:
    s = Path(splits_dir)
    return {
        "train": _read_split(s / "df_train_good.csv", "dataset_public/", pdt_root),
        "val": _read_split(s / "df_val.csv", "dataset_public/", pdt_root),
        "final": _read_split(s / "df_final.csv", "dataset_public/", pdt_root),
    }


def path_exist_rate(df: pd.DataFrame) -> float:
    return float(np.mean([Path(p).exists() for p in df["path"]])) if len(df) else float("nan")


# --------------------------------------------------------------------------------------
# Config decoding (accepts "params_x" or plain "x" keys, as found in CSVs / Optuna DBs)
# --------------------------------------------------------------------------------------
def _get(cfg: dict, key: str):
    if f"params_{key}" in cfg and pd.notna(cfg[f"params_{key}"]):
        return cfg[f"params_{key}"]
    return cfg[key]


def _as_bool(x) -> bool:
    return x if isinstance(x, (bool, np.bool_)) else str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def _key_str(x) -> str:
    """'256.0' -> '256' ; 'True' stays ; floats like 0.875 stay '0.875'."""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def decode_config(cfg: dict, budget: float | None = None) -> dict:
    image_size_key = _key_str(_get(cfg, "image_size_key"))
    layers_key = str(_get(cfg, "layers_key"))
    center_crop_key = _key_str(_get(cfg, "center_crop_key"))
    image_size = IMAGE_SIZE_MAP[image_size_key]
    if center_crop_key == "none":
        crop = None
    elif center_crop_key == "0.875":
        crop = (int(0.875 * image_size[0]), int(0.875 * image_size[0]))
    else:
        raise ValueError(f"Unsupported center_crop_key: {center_crop_key}")
    if budget is None:
        budget = float(_get(cfg, "soft_review_budget"))
    return {
        "backbone": str(_get(cfg, "backbone")),
        "batch_size": int(float(_get(cfg, "batch_size"))),
        "center_crop_key": center_crop_key, "center_crop_size": crop,
        "coreset_sampling_ratio": float(_get(cfg, "coreset_sampling_ratio")),
        "image_size_key": image_size_key, "image_size": image_size,
        "layers_key": layers_key, "layers": LAYERS_MAP[layers_key],
        "max_patches_per_image": int(float(_get(cfg, "max_patches_per_image"))),
        "num_neighbors": int(float(_get(cfg, "num_neighbors"))),
        "pre_trained": _as_bool(_get(cfg, "pre_trained")),
        "reduction": str(_get(cfg, "reduction")),
        "soft_corruption_level": str(_get(cfg, "soft_corruption_level")),
        "soft_review_budget": float(budget),
        "soft_train_fraction": float(_get(cfg, "soft_train_fraction")),
    }


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def _stable_int_seed(text: str, seed: int) -> int:
    return (zlib.crc32(text.encode("utf-8")) + seed) & 0xFFFFFFFF


def corrupt_pil(img: Image.Image, level: str, seed_int: int) -> Image.Image:
    if level == "none":
        return img
    rng = np.random.default_rng(seed_int)
    out = img
    if level in {"mild", "strong"}:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.6 if level == "mild" else 1.2))
    if level == "strong":
        arr = np.asarray(out).astype(np.float32)
        arr = np.clip(arr + rng.normal(0.0, 8.0, size=arr.shape).astype(np.float32), 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
    return out


class ImagePathDataset(Dataset):
    def __init__(self, df, preproc, corruption_level="none", apply_corruption=False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy().astype(int)
        self.preproc, self.level, self.apply = preproc, corruption_level, apply_corruption

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.apply and self.level != "none":
            img = corrupt_pil(img, self.level, _stable_int_seed(self.paths[i], SEED))
        return self.preproc(img), torch.tensor(self.labels[i], dtype=torch.long)


def make_preproc(image_size, crop):
    t = [transforms.Resize(image_size)]
    if crop is not None:
        t.append(transforms.CenterCrop(crop))
    t += [transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
    return transforms.Compose(t)


def make_loader(df, d, corruption=False):
    ds = ImagePathDataset(df, make_preproc(d["image_size"], d["center_crop_size"]),
                          corruption_level=d["soft_corruption_level"], apply_corruption=corruption)
    return DataLoader(ds, batch_size=d["batch_size"], shuffle=False, num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available())


# --------------------------------------------------------------------------------------
# PatchCore
# --------------------------------------------------------------------------------------
def build_backbone(name: str, pre_trained: bool):
    ctor = {"resnet18": (models.resnet18, "ResNet18_Weights"),
            "resnet34": (models.resnet34, "ResNet34_Weights"),
            "resnet50": (models.resnet50, "ResNet50_Weights"),
            "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights")}[name]
    m = ctor[0](weights=getattr(models, ctor[1]).DEFAULT if pre_trained else None)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers, pre_trained):
        super().__init__()
        self.backbone = build_backbone(backbone, pre_trained)
        self.layers = layers
        self._features = {}
        named = dict(self.backbone.named_modules())
        for n in layers:
            named[n].register_forward_hook(self._hook(n))

    def _hook(self, name):
        def fn(module, inp, out):
            self._features[name] = out
        return fn

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)
        return [self._features[n] for n in self.layers]


def to_patch_embeddings(feat_maps, max_patches_per_image=None, seed=0):
    H = max(f.shape[-2] for f in feat_maps)
    W = max(f.shape[-1] for f in feat_maps)
    res = []
    for f in feat_maps:
        if (f.shape[-2], f.shape[-1]) != (H, W):
            f = torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        res.append(f)
    f_cat = torch.cat(res, dim=1)
    _, D, _, _ = f_cat.shape
    f_flat = f_cat.flatten(2).transpose(1, 2).contiguous()
    P = f_flat.shape[1]
    if max_patches_per_image is not None and max_patches_per_image < P:
        g = torch.Generator(device=f_flat.device)
        g.manual_seed(seed)
        idx = torch.randperm(P, generator=g, device=f_flat.device)[:max_patches_per_image]
        f_flat = f_flat[:, idx, :]
    return f_flat.reshape(-1, D)


def coreset_sample(emb: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    n = emb.shape[0]
    m = max(1, int(n * ratio))
    idx = np.random.default_rng(seed).choice(n, size=m, replace=False)
    return emb[idx]


@torch.no_grad()
def fit_memory_bank(extractor, loader, ratio, max_patches, run_seed=SEED):
    extractor = extractor.to(DEVICE).eval()
    kept = []
    for x, _ in loader:
        feats = extractor(x.to(DEVICE, non_blocking=True))
        kept.append(to_patch_embeddings(feats, max_patches, seed=run_seed).detach().cpu().numpy())
    return coreset_sample(np.vstack(kept), ratio=ratio, seed=run_seed)


@torch.no_grad()
def score_images(extractor, loader, memory_bank, k, reduction, max_patches, run_seed=SEED):
    extractor = extractor.to(DEVICE).eval()
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(memory_bank)
    ys, ss = [], []
    for x, y in loader:
        feats = extractor(x.to(DEVICE, non_blocking=True))
        patches = to_patch_embeddings(feats, max_patches, seed=run_seed).detach().cpu().numpy()
        dists, _ = nn.kneighbors(patches, return_distance=True)
        ps = dists.mean(axis=1).reshape(y.shape[0], -1)
        ss.append(ps.max(axis=1) if reduction == "max" else ps.mean(axis=1))
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ss)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def threshold_from_review_budget(score: np.ndarray, budget: float) -> float:
    assert 0.0 < budget < 1.0
    k = max(1, int(np.ceil(budget * score.shape[0])))
    return float(np.partition(score, -k)[-k])


def prf_at_threshold(y, s, thr):
    p, r, f, _ = precision_recall_fscore_support(y, (s >= thr).astype(int), average="binary", zero_division=0)
    return float(p), float(r), float(f)


def safe_auroc(y, s) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def split_metrics(y_val, s_val, y_test, s_test, budget) -> dict:
    """Validation-calibrated review-budget threshold (the paper's deployed rule) + test-calibrated
    variant (what mvtec/01 ``evaluate_best_on_final`` reports)."""
    thr = threshold_from_review_budget(s_val, budget)
    pv, rv, fv = prf_at_threshold(y_val, s_val, thr)
    pt, rt, ft = prf_at_threshold(y_test, s_test, thr)
    thr_t = threshold_from_review_budget(s_test, budget)
    _, _, ft_self = prf_at_threshold(y_test, s_test, thr_t)
    return dict(threshold_val_budget=thr, precision_val=pv, recall_val=rv, f1_val=fv,
                precision_test=pt, recall_test=rt, f1_test=ft, f1_test_self_budget=ft_self,
                auroc_val=safe_auroc(y_val, s_val), auroc_test=safe_auroc(y_test, s_test),
                n_val=int(len(y_val)), n_test=int(len(y_test)),
                prevalence_val=float(np.mean(y_val)), prevalence_test=float(np.mean(y_test)))


def f1_perfect_ranker(prev: float, budget: float) -> float:
    """F1 of a *perfect* detector flagging a fraction ``budget`` when the positive rate is ``prev``."""
    return 2 * min(budget, prev) / (budget + prev)


def f1_random_ranker(prev: float, budget: float) -> float:
    return 2 * prev * budget / (prev + budget)


# --------------------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------------------
def _subsample(df_train: pd.DataFrame, frac: float, seed_offset: int, run_seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df_train
    return df_train.sample(frac=frac, random_state=run_seed + seed_offset)


def fit_and_score(d: dict, df_train: pd.DataFrame, eval_sets: dict, seed_offset=0, run_seed=SEED) -> dict:
    """Fit one memory bank on ``df_train`` and score every frame in ``eval_sets``.
    Returns {"scores": {name: (y, s)}, "meta": {...}}."""
    t0 = time.time()
    df_sub = _subsample(df_train, d["soft_train_fraction"], seed_offset, run_seed)
    # (The original notebooks seed torch once per kernel, so a pre_trained=False backbone gets a
    #  different random init on every call. Seeding per run removes that uncontrolled variance
    #  without changing any modelling choice. Pretrained weights are unaffected.)
    torch.manual_seed(run_seed + seed_offset)

    extractor = FeatureExtractor(d["backbone"], d["layers"], d["pre_trained"])
    bank = fit_memory_bank(extractor, make_loader(df_sub, d, corruption=True),
                           d["coreset_sampling_ratio"], d["max_patches_per_image"], run_seed)
    if len(bank) < d["num_neighbors"]:
        raise ValueError(f"memory bank {len(bank)} < k={d['num_neighbors']}")
    scores = {}
    for name, df in eval_sets.items():
        scores[name] = score_images(extractor, make_loader(df, d), bank, d["num_neighbors"],
                                    d["reduction"], d["max_patches_per_image"], run_seed)
    meta = dict(train_good_used=int(len(df_sub)), memory_bank_size=int(bank.shape[0]),
                runtime_sec=float(time.time() - t0))
    del extractor, bank
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"scores": scores, "meta": meta}


def run_single(cfg: dict, data: dict, budget=None, seed_offset=0, run_seed=SEED) -> dict:
    """One model, one threshold. Used for Proditec and for the *pooled* MVTec protocol of
    mvtec/05_confirmation_evaluation.ipynb (``data`` = pool_class_data(...))."""
    d = decode_config(cfg, budget)
    r = fit_and_score(d, data["train"], {"val": data["val"], "test": data["final"]}, seed_offset, run_seed)
    (yv, sv), (yt, st) = r["scores"]["val"], r["scores"]["test"]
    out = split_metrics(yv, sv, yt, st, d["soft_review_budget"])
    out.update(r["meta"])
    out["_scores"] = r["scores"]
    return out


def run_per_category(cfg: dict, class_data: dict, budget=None, seed_offset=0, run_seed=SEED) -> dict:
    """One PatchCore + one review-budget threshold *per category* (the MVTec HPO objective).
    Also derives, from the very same models, the pooled-threshold / pooled-AUROC view."""
    d = decode_config(cfg, budget)
    b = d["soft_review_budget"]
    per, raw, runtime = {}, {}, 0.0
    for cls, parts in class_data.items():
        r = fit_and_score(d, parts["train"], {"val": parts["val"], "test": parts["final"]}, seed_offset, run_seed)
        (yv, sv), (yt, st) = r["scores"]["val"], r["scores"]["test"]
        per[cls] = split_metrics(yv, sv, yt, st, b)
        per[cls].update(r["meta"])
        raw[cls] = r["scores"]
        runtime += r["meta"]["runtime_sec"]
    agg = {f"mean_{k}": float(np.mean([per[c][k] for c in per]))
           for k in ("f1_val", "f1_test", "f1_test_self_budget", "auroc_val", "auroc_test",
                     "precision_test", "recall_test")}
    # pooled view of the SAME models (single threshold over merged scores)
    yv = np.concatenate([raw[c]["val"][0] for c in raw]); sv = np.concatenate([raw[c]["val"][1] for c in raw])
    yt = np.concatenate([raw[c]["test"][0] for c in raw]); st = np.concatenate([raw[c]["test"][1] for c in raw])
    pooled = split_metrics(yv, sv, yt, st, b)
    out = dict(per_class=per, **agg,
               pooled_f1_val=pooled["f1_val"], pooled_f1_test=pooled["f1_test"],
               pooled_auroc_test=pooled["auroc_test"], runtime_sec=float(runtime),
               prevalence_test=pooled["prevalence_test"], budget=b)
    out["_scores"] = raw
    return out


def flatten_per_category(res: dict, prefix="") -> dict:
    """Scalar view of a ``run_per_category`` result (for DataFrames)."""
    row = {f"{prefix}{k}": v for k, v in res.items() if k not in ("per_class", "_scores")}
    for cls, m in res["per_class"].items():
        for k in ("f1_val", "f1_test", "auroc_test"):
            row[f"{prefix}{cls}_{k}"] = m[k]
    return row


def flatten_single(res: dict, prefix="") -> dict:
    return {f"{prefix}{k}": v for k, v in res.items() if k != "_scores" and not isinstance(v, (dict, np.ndarray))}


# --------------------------------------------------------------------------------------
# Caching / resume
# --------------------------------------------------------------------------------------
def cache_key(*objs) -> str:
    return hashlib.md5(json.dumps(list(objs), sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached(cache_dir: Path, key: str, fn, keep_scores=False):
    """Run ``fn()`` once; later calls with the same key load the pickle (resume after a crash)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    res = fn()
    if not keep_scores:
        res = {k: v for k, v in res.items() if k != "_scores"}
    with open(p, "wb") as f:
        pickle.dump(res, f)
    return res


# --------------------------------------------------------------------------------------
# Optuna DB reader (no optuna dependency needed)
# --------------------------------------------------------------------------------------
def load_hpo_trials(db_path: Path, study_name_contains: str | None = None) -> pd.DataFrame:
    """Decoded COMPLETE trials of one study: number, value, <param>, runtime_sec, user attrs."""
    import sqlite3
    con = sqlite3.connect(str(db_path))
    studies = pd.read_sql("select study_id, study_name from studies", con)
    if study_name_contains:
        studies = studies[studies.study_name.str.contains(study_name_contains, regex=False)]
    sid = int(studies.sort_values("study_id").iloc[-1]["study_id"])
    tr = pd.read_sql(f"select trial_id, number, state from trials where study_id={sid} and state='COMPLETE'", con)
    val = pd.read_sql("select trial_id, value from trial_values where objective=0", con)
    prm = pd.read_sql("select trial_id, param_name, param_value, distribution_json from trial_params", con)
    prm = prm[prm.trial_id.isin(tr.trial_id)]

    def dec(r):
        dist = json.loads(r.distribution_json)
        ch = dist.get("attributes", {}).get("choices")
        return ch[int(r.param_value)] if ch is not None else r.param_value

    prm["v"] = prm.apply(dec, axis=1)
    wide = prm.pivot(index="trial_id", columns="param_name", values="v").reset_index()
    ua = pd.read_sql("select trial_id, key, value_json from trial_user_attributes", con)
    ua = ua[ua.trial_id.isin(tr.trial_id) & ua.key.isin(["runtime_sec", "mean_f1_val_at_budget", "f1_val_at_budget"])]
    ua["v"] = ua.value_json.map(json.loads)
    uaw = ua.pivot(index="trial_id", columns="key", values="v").reset_index()
    out = tr.merge(val, on="trial_id").merge(wide, on="trial_id").merge(uaw, on="trial_id", how="left")
    con.close()
    return out.sort_values("number").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Uniform runner / helpers shared by R01-R03
# --------------------------------------------------------------------------------------
def smoke_subsample(data: dict, n: int = 24, seed: int = SEED) -> dict:
    """Keep <= n images per split (stratified by label). Plumbing tests only."""
    def sub(df):
        if len(df) <= n:
            return df
        parts = [g.sample(n=max(1, int(round(n * len(g) / len(df)))), random_state=seed) for _, g in df.groupby("label")]
        return pd.concat(parts).reset_index(drop=True)
    return {k: sub(v) for k, v in data.items()}


def run_dataset(ds: str, cfg: dict, data, budget=None, seed_offset=0, run_seed=SEED) -> dict:
    """ds='mvtec': ``data`` = {class: {train,val,final}} -> per-category protocol (matches the HPO objective).
    ds='proditec': ``data`` = {train,val,final} -> single model. Returns a result incl. raw ``_scores``."""
    if ds == "mvtec":
        return run_per_category(cfg, data, budget, seed_offset, run_seed)
    return run_single(cfg, data, budget, seed_offset, run_seed)


def summarise(ds: str, res: dict) -> dict:
    """Uniform scalar summary of ``run_dataset`` output (what the surrogate predicts = validation F1)."""
    if ds == "mvtec":
        row = dict(f1_val=res["mean_f1_val"], f1_test=res["mean_f1_test"], auroc_val=res["mean_auroc_val"],
                   auroc_test=res["mean_auroc_test"], pooled_f1_test=res["pooled_f1_test"],
                   pooled_auroc_test=res["pooled_auroc_test"], prevalence_test=res["prevalence_test"],
                   runtime_sec=res["runtime_sec"], budget=res["budget"])
        for c, m in res["per_class"].items():
            row[f"{c}_f1_val"], row[f"{c}_f1_test"], row[f"{c}_auroc_test"] = m["f1_val"], m["f1_test"], m["auroc_test"]
        return row
    return dict(f1_val=res["f1_val"], f1_test=res["f1_test"], auroc_val=res["auroc_val"], auroc_test=res["auroc_test"],
                prevalence_test=res["prevalence_test"], runtime_sec=res["runtime_sec"])


def sweep_budget(ds: str, raw_scores: dict, budgets) -> pd.DataFrame:
    """Re-threshold already computed scores at many review budgets (scores do not depend on the budget).
    mvtec: ``raw_scores`` = {class: {val:(y,s), test:(y,s)}}; proditec: {val:(y,s), test:(y,s)}."""
    rows = []
    for b in budgets:
        if ds == "mvtec":
            per = {c: split_metrics(*v["val"], *v["test"], b) for c, v in raw_scores.items()}
            yv = np.concatenate([v["val"][0] for v in raw_scores.values()]); sv = np.concatenate([v["val"][1] for v in raw_scores.values()])
            yt = np.concatenate([v["test"][0] for v in raw_scores.values()]); st = np.concatenate([v["test"][1] for v in raw_scores.values()])
            pooled = split_metrics(yv, sv, yt, st, b)
            prev_c = {c: per[c]["prevalence_test"] for c in per}
            rows.append(dict(budget=b, f1_val=np.mean([m["f1_val"] for m in per.values()]),
                             f1_test=np.mean([m["f1_test"] for m in per.values()]),
                             pooled_f1_test=pooled["f1_test"], pooled_f1_val=pooled["f1_val"],
                             perfect=np.mean([f1_perfect_ranker(p, b) for p in prev_c.values()]),
                             random=np.mean([f1_random_ranker(p, b) for p in prev_c.values()]),
                             pooled_perfect=f1_perfect_ranker(pooled["prevalence_test"], b),
                             pooled_random=f1_random_ranker(pooled["prevalence_test"], b),
                             auroc_test=np.mean([m["auroc_test"] for m in per.values()]),
                             pooled_auroc_test=pooled["auroc_test"]))
        else:
            m = split_metrics(*raw_scores["val"], *raw_scores["test"], b)
            rows.append(dict(budget=b, f1_val=m["f1_val"], f1_test=m["f1_test"],
                             perfect=f1_perfect_ranker(m["prevalence_test"], b),
                             random=f1_random_ranker(m["prevalence_test"], b), auroc_test=m["auroc_test"]))
    return pd.DataFrame(rows)


def cfg_signature(cfg: dict) -> dict:
    """Keys that define a configuration (for cache keys)."""
    keys = ["backbone", "batch_size", "center_crop_key", "image_size_key", "layers_key", "max_patches_per_image",
            "pre_trained", "reduction", "soft_corruption_level", "coreset_sampling_ratio", "num_neighbors",
            "soft_review_budget", "soft_train_fraction"]
    out = {}
    for k in keys:
        try:
            out[k] = str(_get(cfg, k))
        except KeyError:
            pass
    return out


def f1_at_prevalence(y, s, thr, prevalence: float) -> float:
    """F1 the same detector/threshold would obtain if the test set had positive rate ``prevalence``
    (negatives and positives re-weighted; removes the MVTec-vs-Proditec prevalence confound)."""
    y = np.asarray(y).astype(int); s = np.asarray(s)
    pos, neg = y == 1, y == 0
    rec = float(np.mean(s[pos] >= thr)) if pos.any() else 0.0
    fpr = float(np.mean(s[neg] >= thr)) if neg.any() else 0.0
    tp, fp = prevalence * rec, (1 - prevalence) * fpr
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def _data_fingerprint(ds: str, data) -> list:
    if ds == "mvtec":
        return [[c, len(v["train"]), len(v["val"]), len(v["final"])] for c, v in data.items()]
    return [len(data["train"]), len(data["val"]), len(data["final"])]


def run_many(ds: str, cfgs: list, data, seeds=(SEED,), cache_dir=None, ids=None, offsets=None,
             budget_override=None, verbose=True):
    """Run many configurations x seeds with on-disk caching (a crashed notebook resumes where it stopped).

    cfgs     : list of config dicts (keys with or without the ``params_`` prefix)
    ids      : label per cfg (default 0..n-1);  offsets : ``seed_offset`` per cfg (HPO used the trial number,
               the confirmation notebooks used cf_id)
    returns  : (DataFrame of scalar summaries, {(id, seed): raw scores})
    """
    ids = list(range(len(cfgs))) if ids is None else list(ids)
    offsets = ids if offsets is None else list(offsets)
    fp = _data_fingerprint(ds, data)
    rows, raw = [], {}
    t0 = time.time()
    for i, (cid, cfg, off) in enumerate(zip(ids, cfgs, offsets)):
        for sd in seeds:
            sig = cfg_signature(cfg)
            if budget_override is not None:
                sig["soft_review_budget"] = str(budget_override)
            key = cache_key(ds, sig, int(off), int(sd), fp)
            res = cached(cache_dir, key, lambda: run_dataset(ds, cfg, data, budget_override, int(off), int(sd)),
                         keep_scores=True)
            row = dict(id=cid, seed=int(sd), seed_offset=int(off), **summarise(ds, res))
            row.setdefault("budget", float(sig["soft_review_budget"]))
            rows.append(row)
            raw[(cid, int(sd))] = res.get("_scores")
        if verbose:
            print(f"  [{i + 1}/{len(cfgs)}] id={cid}  elapsed {time.time() - t0:7.1f}s  "
                  f"f1_val={rows[-1]['f1_val']:.3f}  f1_test={rows[-1]['f1_test']:.3f}")
    return pd.DataFrame(rows), raw
