"""
F1 Pit-Stop 预测 —— CatBoost 主模型 + MLP（残差版）+ Optuna 调优
Kaggle Playground Series S6E5 | 目标列 PitNextLap (二分类, AUC)

修改内容：
  - TabMLP 内部使用 ResidualBlock 替代普通全连接层
  - 其他所有部分与原版一致（特征工程、CatBoost、集成策略）

BUG 修复：
  1. [严重] 类别列是 str，MLP 分支直接 to_numpy(int64) 会崩溃
     → 新增 LabelEncoder，MLP 分支使用编码后的整数列
  2. [高]   多 seed 下 OOF 累加/除法逻辑错误（不同 seed 切分不同）
     → 改用 oof_count 数组记录每个样本被预测次数，逐元素相除
  3. [中]   测试集 MLP 推理一次性送入可能 OOM
     → 测试集也分批推理
  4. [中]   submission 用 merge 可能引入 NaN
     → 改为直接赋值
  5. [低]   Embedding num_embeddings 无安全余量
     → +1 防止索引越界
"""
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder          # [FIX-1] 新增
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
TARGET  = "PitNextLap"
N_FOLDS = 5
SEEDS   = [2026]          # 可增加 [2026, 7, 77]
CAT_COLS = ["Driver", "Compound", "Race"]

N_TRIALS_CB = 20
N_TRIALS_NN = 15
CB_ITERS = 3000
NN_EPOCHS = 50

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "playground-series-s6e5"

# ===================== 读数据 =====================
train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw  = pd.read_csv(DATA_DIR / "test.csv")
subm      = pd.read_csv(DATA_DIR / "sample_submission.csv")
print(f"Train: {train_raw.shape}  Test: {test_raw.shape}  Pos rate: {train_raw[TARGET].mean():.4f}")

y = train_raw[TARGET].values.astype(int)
POS_W = (y == 0).sum() / max((y == 1).sum(), 1)

# ===================== 特征工程（全部保留） =====================
def features(df):
    """行级基础特征 + 新增33个物理/策略特征"""
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype(str)

    # 原有基础特征（35个）
    df["TyreLife_sq"]  = df["TyreLife"] ** 2
    df["TyreLife_cb"]  = df["TyreLife"] ** 3
    df["TyreLife_log"] = np.log1p(df["TyreLife"])
    df["LapNumber_sq"] = df["LapNumber"] ** 2
    df["LapNumber_log"] = np.log1p(df["LapNumber"])
    df["LapTime_log"]  = np.log1p(df["LapTime (s)"].clip(lower=0))
    df["Deg_per_lap"]  = df["Cumulative_Degradation"] / (df["TyreLife"] + 1)
    df["Deg_x_TyreLife"] = df["Cumulative_Degradation"] * df["TyreLife"]
    df["Deg_x_Progress"] = df["Cumulative_Degradation"] * df["RaceProgress"]
    df["AbsDeg"]       = df["Cumulative_Degradation"].abs()
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]
    df["Stint_x_Progress"] = df["Stint"] * df["RaceProgress"]
    df["DeltaAbs"]     = df["LapTime_Delta"].abs()
    df["Delta_sq"]     = df["LapTime_Delta"] ** 2
    df["LapTime_x_TyreLife"] = df["LapTime (s)"] * df["TyreLife"]
    df["LapTime_x_Position"] = df["LapTime (s)"] * df["Position"]
    df["Progress_sq"]  = df["RaceProgress"] ** 2
    df["LapsRemaining"] = 1.0 - df["RaceProgress"]
    df["Progress_x_TyreLife"] = df["RaceProgress"] * df["TyreLife"]
    df["Progress_x_Stint"]    = df["RaceProgress"] * df["Stint"]
    df["Position_inv"] = 1.0 / (df["Position"] + 1)
    df["PosChange_abs"] = df["Position_Change"].abs()
    df["Position_x_Progress"] = df["Position"] * df["RaceProgress"]
    df["TyreLife_x_Delta"]    = df["TyreLife"] * df["LapTime_Delta"]
    df["IsEarly"]  = (df["RaceProgress"] < 0.25).astype("int8")
    df["IsMid"]    = ((df["RaceProgress"] >= 0.25) & (df["RaceProgress"] < 0.75)).astype("int8")
    df["IsLate"]   = (df["RaceProgress"] >= 0.75).astype("int8")
    df["IsLead"]   = (df["Position"] == 1).astype("int8")
    df["IsTop5"]   = (df["Position"] <= 5).astype("int8")
    df["IsStint1"] = (df["Stint"] == 1).astype("int8")
    df["IsPit0"]   = (df["PitStop"] == 0).astype("int8")
    df["feat_WearRate"]         = df["TyreLife"] / (df["LapNumber"] + 1)
    df["feat_TyreLife_div_Lap"] = df["TyreLife"] / df["LapNumber"].clip(lower=1)
    df["feat_Pos_per_Lap"]      = df["Position"] / df["LapNumber"].clip(lower=1)
    df["feat_Lap_div_Progress"] = df["LapNumber"] / (df["RaceProgress"] + 1e-6)

    # 新增 Part A：33个行级物理/策略特征
    eps = 1e-6
    D  = df["Cumulative_Degradation"]
    TL = df["TyreLife"]
    LR = df["LapsRemaining"]
    P  = df["RaceProgress"]
    LT = df["LapTime (s)"]

    df["Deg_sq"]   = D ** 2
    df["Deg_cube"] = D ** 3
    df["Deg_per_lap_sq"] = df["Deg_per_lap"] ** 2
    df["WearLoad"] = TL * df["Deg_per_lap"]
    df["TyreLife_x_Deg_x_Progress"] = TL * D * P
    df["Deg_x_LapsRemaining"]  = D * LR
    df["Deg_div_LapsRemaining"] = D / (LR + eps)
    df["Deg_x_Position_inv"]   = D * df["Position_inv"]

    df["DistToHalf"] = (P - 0.5).abs()
    pv = P.to_numpy()
    df["PitWindow_score"] = np.maximum.reduce(
        [np.exp(-((pv - c) ** 2) / (2 * s * s)) for c, s in [(1/3, 0.08), (0.5, 0.10), (2/3, 0.08)]])
    df["Deg_in_early"] = D * df["IsEarly"]
    df["OldTyre_late"] = TL * df["IsLate"]

    df["Delta_x_Progress"]   = df["LapTime_Delta"] * P
    df["Delta_per_TyreLife"] = df["LapTime_Delta"] / (TL + 1)
    df["Delta_x_Deg"]        = df["LapTime_Delta"] * D
    df["LapTime_x_Progress"] = LT * P
    df["LapTime_per_TyreLife"] = LT / (TL + 1)

    df["Position_sq"]  = df["Position"] ** 2
    df["Position_log"] = np.log1p(df["Position"])
    df["PosChange_pos"] = df["Position_Change"].clip(lower=0)
    df["PosChange_neg"] = df["Position_Change"].clip(upper=0)
    df["PosChange_x_Progress"] = df["Position_Change"] * P
    df["PosChange_x_Deg"]      = df["Position_Change"] * D
    df["InPoints"] = (df["Position"] <= 10).astype("int8")
    df["Podium"]   = (df["Position"] <= 3).astype("int8")

    df["Stint_sq"] = df["Stint"] ** 2
    df["IsFreshStint"]       = (TL <= 2).astype("int8")
    df["Stint_x_Deg"]        = df["Stint"] * D
    df["TyreLife_per_Stint"] = TL / df["Stint"].clip(lower=1)

    df["PitStop_sq"] = df["PitStop"] ** 2
    df["HasPitted"]  = (df["PitStop"] >= 1).astype("int8")
    df["PitRate"]    = df["PitStop"] / (df["LapNumber"] + 1)
    df["PitStop_x_Progress"] = df["PitStop"] * P

    return df.replace([np.inf, -np.inf], np.nan)


def add_relative_features(tr_df, te_df):
    """新增 Part B：33个相对/策略特征（train+test联合计算，无标签泄露）"""
    eps = 1e-6
    n_tr = len(tr_df)
    both = pd.concat([tr_df, te_df], axis=0, ignore_index=True)
    both["_row"] = np.arange(len(both))

    # (G) 同圈场内相对
    g = both.groupby(["Race", "LapNumber"])
    for col, nm in [("LapTime (s)", "LapTime"), ("Cumulative_Degradation", "Deg"), ("TyreLife", "TyreLife")]:
        m = g[col].transform("mean")
        s = g[col].transform("std").replace(0, np.nan)
        both[f"rel_{nm}_lap_z"]    = (both[col] - m) / (s + eps)
        both[f"rel_{nm}_lap_rank"] = g[col].rank(pct=True)
    both["rel_field_size"]     = both.groupby(["Race", "LapNumber"])["LapNumber"].transform("size")
    both["rel_field_pace_std"] = g["LapTime (s)"].transform("std")
    both["rel_field_deg_mean"] = g["Cumulative_Degradation"].transform("mean")
    both["rel_deg_vs_field"]   = both["Cumulative_Degradation"] - both["rel_field_deg_mean"]
    both["rel_pos_pct"]        = both["Position"] / both["rel_field_size"].clip(lower=1)

    # (H) 相邻车
    both = both.sort_values(["Race", "LapNumber", "Position"])
    gp = both.groupby(["Race", "LapNumber"], sort=False)
    tl_ahead = gp["TyreLife"].shift(1);    tl_behind = gp["TyreLife"].shift(-1)
    lt_ahead = gp["LapTime (s)"].shift(1); lt_behind = gp["LapTime (s)"].shift(-1)
    cp_ahead = gp["Compound"].shift(1);    cp_behind = gp["Compound"].shift(-1)
    both["rel_dTyreLife_ahead"]  = both["TyreLife"] - tl_ahead
    both["rel_dTyreLife_behind"] = both["TyreLife"] - tl_behind
    both["rel_pace_vs_ahead"]    = lt_ahead - both["LapTime (s)"]
    both["rel_pace_vs_behind"]   = both["LapTime (s)"] - lt_behind
    both["rel_same_comp_ahead"]  = (both["Compound"] == cp_ahead).astype("int8")
    both["rel_same_comp_behind"] = (both["Compound"] == cp_behind).astype("int8")
    both["rel_undercut_threat"]  = ((tl_behind < both["TyreLife"]) & (both["rel_pace_vs_behind"] > 0)).astype("int8")
    both["rel_overcut_setup"]    = ((tl_ahead  > both["TyreLife"]) & (both["rel_pace_vs_ahead"] <= 0)).astype("int8")

    # (I) 化合物寿命窗口
    gc = both.groupby(["Race", "Compound"])
    both["rel_comp_life"]        = gc["TyreLife"].transform(lambda x: x.quantile(0.95))
    both["rel_tyrelife_vs_comp"] = both["TyreLife"] / (both["rel_comp_life"] + eps)
    both["rel_comp_life_left"]   = both["rel_comp_life"] - both["TyreLife"]

    # (J) 比赛剩余 / 停站预算
    gr = both.groupby("Race")
    both["rel_race_total_laps"] = gr["LapNumber"].transform("max")
    both["rel_laps_remaining"]  = both["rel_race_total_laps"] - both["LapNumber"]
    both["rel_is_final_laps"]   = (both["rel_laps_remaining"] <= 5).astype("int8")
    both["rel_can_amortize"]    = (both["rel_laps_remaining"] >= 8).astype("int8")
    both["rel_race_max_stops"]  = gr["PitStop"].transform("max")
    both["rel_stops_left"]      = both["rel_race_max_stops"] - both["PitStop"]

    # (K) 时间趋势
    both = both.sort_values(["Race", "Driver", "LapNumber"])
    col_lt = both.groupby(["Race", "Driver"], sort=False)["LapTime (s)"]
    both["rel_laptime_trend"]  = col_lt.diff()
    both["rel_delta_trend"]    = both.groupby(["Race", "Driver"], sort=False)["LapTime_Delta"].diff()
    both["rel_deg_marginal"]   = both.groupby(["Race", "Driver"], sort=False)["Cumulative_Degradation"].diff()
    both["rel_deg_accel"]      = both.groupby(["Race", "Driver"], sort=False)["rel_deg_marginal"].diff()
    roll3 = col_lt.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    both["rel_laptime_trend3"] = both["LapTime (s)"] - roll3

    # (L) 队友配合
    team_col = next((c for c in ["Team", "Constructor", "team", "constructor"] if c in both.columns), None)
    if team_col is not None:
        gt = both.groupby(["Race", "LapNumber", team_col])
        cnt = gt["TyreLife"].transform("size")
        mate_tl = (gt["TyreLife"].transform("sum") - both["TyreLife"]) / (cnt - 1).clip(lower=1)
        both["rel_team_tyre_gap"]   = np.where(cnt > 1, both["TyreLife"] - mate_tl, np.nan)
        both["rel_team_mate_fresh"] = np.where(cnt > 1, (mate_tl < both["TyreLife"]).astype(float), np.nan)

    both = both.sort_values("_row").drop(columns=["_row"])
    both = both.replace([np.inf, -np.inf], np.nan)
    tr_out = both.iloc[:n_tr].reset_index(drop=True)
    te_out = both.iloc[n_tr:].reset_index(drop=True)
    return tr_out, te_out


# ===================== 应用特征工程 =====================
train_fe = features(train_raw)
test_fe  = features(test_raw)
train_fe, test_fe = add_relative_features(train_fe, test_fe)

FEAT_COLS = [c for c in train_fe.columns if c not in ["id", TARGET]]
NUM_FEATS = [c for c in FEAT_COLS if c not in CAT_COLS]
print(f"总特征数: {len(FEAT_COLS)} (数值: {len(NUM_FEATS)}, 类别: {CAT_COLS})")

# ---------- CatBoost 用的 DataFrame（保持 str 类别） ----------
X_train_cb = train_fe[FEAT_COLS].copy()
X_test_cb  = test_fe[FEAT_COLS].copy()
for c in CAT_COLS:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_test_cb[c]  = X_test_cb[c].astype(str)

# ---------- [FIX-1] MLP 用的 DataFrame（LabelEncoder 编码类别） ----------
label_encoders = {}
X_train_mlp = train_fe[FEAT_COLS].copy()
X_test_mlp  = test_fe[FEAT_COLS].copy()
for c in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([X_train_mlp[c].astype(str), X_test_mlp[c].astype(str)]))
    X_train_mlp[c] = le.transform(X_train_mlp[c].astype(str))
    X_test_mlp[c]  = le.transform(X_test_mlp[c].astype(str))
    label_encoders[c] = le

# 数值特征（用于 MLP，需要填充 NaN 和标准化）
num_df = train_fe[NUM_FEATS].copy()
num_test = test_fe[NUM_FEATS].copy()
imputer = SimpleImputer(strategy="median")
num_arr = imputer.fit_transform(num_df)
num_test_arr = imputer.transform(num_test)

X_num_all = num_arr
X_num_test = num_test_arr

# ===================== Optuna 调优 CatBoost =====================
def objective_cb(trial):
    params = {
        "iterations": CB_ITERS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 255),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "scale_pos_weight": POS_W,
        "thread_count": -1,
        "early_stopping_rounds": 200,
        "verbose": False,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X_train_cb, y):
        X_tr, X_va = X_train_cb.iloc[tr_idx], X_train_cb.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = CatBoostClassifier(**params, random_seed=42)
        model.fit(Pool(X_tr, y_tr, cat_features=CAT_COLS),
                  eval_set=Pool(X_va, y_va, cat_features=CAT_COLS),
                  verbose=False)
        p = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, p))
    return np.mean(aucs)

print("\n=== Optuna 调优 CatBoost ===")
study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026))
study_cb.optimize(objective_cb, n_trials=N_TRIALS_CB, show_progress_bar=True)
best_cb_params = study_cb.best_params
best_cb_params.update({"iterations": CB_ITERS, "loss_function": "Logloss",
                       "eval_metric": "AUC", "scale_pos_weight": POS_W,
                       "thread_count": -1, "early_stopping_rounds": 200, "verbose": False})
print(f"CatBoost 最佳参数: {best_cb_params}")
print(f"最佳 AUC (3折平均): {study_cb.best_value:.5f}")

# ===================== PyTorch MLP（残差版） =====================
USE_NN = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    USE_NN = True
except ImportError:
    print("PyTorch 未安装，将跳过 MLP 模型。")

if USE_NN:
    # [FIX-5] Embedding 容量 = 类别数 + 1，防止索引越界
    cat_cards = {}
    for c in CAT_COLS:
        cat_cards[c] = int(X_train_mlp[c].max()) + 1   # 已经是 0-based 整数
    emb_dims = [(cat_cards[c] + 1, min(50, (cat_cards[c] + 1) // 2)) for c in CAT_COLS]

    # ----- 残差块 -----
    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            )
            self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.block(x) + self.shortcut(x))

    # ----- TabMLP（残差版） -----
    class TabMLP(nn.Module):
        def __init__(self, n_num, emb_dims, hidden_dims, dropout):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in emb_dims])
            self.bn_num = nn.BatchNorm1d(n_num)
            inp = n_num + sum(d for _, d in emb_dims)

            blocks = []
            current = inp
            for h in hidden_dims:
                blocks.append(ResidualBlock(current, h, dropout))
                current = h
            blocks.append(nn.Linear(current, 1))
            self.mlp = nn.Sequential(*blocks)

        def forward(self, xnum, xcat):
            e = [emb(xcat[:, i]) for i, emb in enumerate(self.embs)]
            x = torch.cat([self.bn_num(xnum)] + e, dim=1)
            return self.mlp(x).squeeze(1)

    # [FIX-3] 分批推理辅助函数
    def predict_in_batches(net, num_tensor, cat_tensor, device, batch_size=8192):
        """对任意数据集分批推理，避免 OOM"""
        net.eval()
        preds = []
        n = num_tensor.shape[0]
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb_n = num_tensor[start:end].to(device)
                xb_c = cat_tensor[start:end].to(device)
                preds.append(torch.sigmoid(net(xb_n, xb_c)).cpu().numpy())
        return np.concatenate(preds)

    def objective_nn(trial):
        hidden_dims = [trial.suggest_int(f"h{i}", 32, 256, step=32) for i in range(trial.suggest_int("n_layers", 1, 3))]
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 512, 4096, step=512)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num_all, y)):
            tr_num, va_num = X_num_all[tr_idx], X_num_all[va_idx]
            mean, std = tr_num.mean(axis=0), tr_num.std(axis=0) + 1e-6
            tr_num_norm = (tr_num - mean) / std
            va_num_norm = (va_num - mean) / std

            # [FIX-1] 使用已经 LabelEncoded 的 X_train_mlp
            tr_cat = X_train_mlp.iloc[tr_idx][CAT_COLS].to_numpy(np.int64)
            va_cat = X_train_mlp.iloc[va_idx][CAT_COLS].to_numpy(np.int64)

            tr_ds = TensorDataset(torch.tensor(tr_num_norm, dtype=torch.float32),
                                  torch.tensor(tr_cat, dtype=torch.long),
                                  torch.tensor(y[tr_idx], dtype=torch.float32))
            va_ds = TensorDataset(torch.tensor(va_num_norm, dtype=torch.float32),
                                  torch.tensor(va_cat, dtype=torch.long),
                                  torch.tensor(y[va_idx], dtype=torch.float32))
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=8192, shuffle=False)

            net = TabMLP(len(NUM_FEATS), emb_dims, hidden_dims, dropout).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_W], device=device))

            best_auc = -1
            patience = 5
            bad = 0
            for ep in range(NN_EPOCHS):
                net.train()
                for xb_n, xb_c, yb in tr_loader:
                    xb_n, xb_c, yb = xb_n.to(device), xb_c.to(device), yb.to(device)
                    opt.zero_grad()
                    loss_fn(net(xb_n, xb_c), yb).backward()
                    opt.step()

                net.eval()
                preds = []
                with torch.no_grad():
                    for xb_n, xb_c, _ in va_loader:
                        xb_n, xb_c = xb_n.to(device), xb_c.to(device)
                        preds.append(torch.sigmoid(net(xb_n, xb_c)).cpu().numpy())
                p = np.concatenate(preds)
                auc = roc_auc_score(y[va_idx], p)
                if auc > best_auc:
                    best_auc = auc
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            aucs.append(best_auc)
        return np.mean(aucs)

    print("\n=== Optuna 调优 PyTorch MLP（残差版） ===")
    study_nn = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026))
    study_nn.optimize(objective_nn, n_trials=N_TRIALS_NN, show_progress_bar=True)
    best_nn_params = study_nn.best_params
    print(f"MLP 最佳参数: {best_nn_params}")
    print(f"最佳 AUC (3折平均): {study_nn.best_value:.5f}")
else:
    best_nn_params = None

# ===================== 交叉验证训练最终模型 =====================

# [FIX-2] CatBoost 多 seed OOF：用 count 数组正确平均
def train_catboost_cv(params, seeds):
    oof = np.zeros(len(y), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int32)       # 记录每个样本被预测次数
    test_pred = np.zeros(len(test_raw), dtype=np.float64)
    aucs = []
    n_runs = len(seeds) * N_FOLDS
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_cb, y), 1):
            X_tr, X_va = X_train_cb.iloc[tr_idx], X_train_cb.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model = CatBoostClassifier(**params, random_seed=seed)
            model.fit(Pool(X_tr, y_tr, cat_features=CAT_COLS),
                      eval_set=Pool(X_va, y_va, cat_features=CAT_COLS),
                      verbose=False)
            p_va = model.predict_proba(X_va)[:, 1]
            oof[va_idx] += p_va
            oof_count[va_idx] += 1                      # [FIX-2]
            test_pred += model.predict_proba(X_test_cb)[:, 1] / n_runs
            auc = roc_auc_score(y_va, p_va)
            aucs.append(auc)
            print(f"  seed{seed} fold{fold} | CB AUC={auc:.5f}")
    oof /= np.maximum(oof_count, 1)                     # [FIX-2] 逐元素除以实际预测次数
    return oof, test_pred, np.mean(aucs), np.std(aucs)


# [FIX-2] MLP 多 seed OOF：同样使用 count 数组
# [FIX-1] 使用 X_train_mlp / X_test_mlp（已编码）
# [FIX-3] 测试集分批推理
def train_mlp_cv(params, seeds):
    if not USE_NN:
        return None, None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.zeros(len(y), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int32)        # [FIX-2]
    test_pred = np.zeros(len(test_raw), dtype=np.float64)
    aucs = []
    n_runs = len(seeds) * N_FOLDS
    hidden_dims = [params[f"h{i}"] for i in range(params["n_layers"])]
    dropout = params["dropout"]
    lr = params["lr"]
    wd = params["wd"]
    batch_size = params["batch_size"]

    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num_all, y), 1):
            tr_num, va_num = X_num_all[tr_idx], X_num_all[va_idx]
            mean, std = tr_num.mean(axis=0), tr_num.std(axis=0) + 1e-6
            tr_num_norm = (tr_num - mean) / std
            va_num_norm = (va_num - mean) / std
            te_num_norm = (X_num_test - mean) / std

            # [FIX-1] 使用已编码的 X_train_mlp / X_test_mlp
            tr_cat = X_train_mlp.iloc[tr_idx][CAT_COLS].to_numpy(np.int64)
            va_cat = X_train_mlp.iloc[va_idx][CAT_COLS].to_numpy(np.int64)
            te_cat = X_test_mlp[CAT_COLS].to_numpy(np.int64)

            tr_ds = TensorDataset(torch.tensor(tr_num_norm, dtype=torch.float32),
                                  torch.tensor(tr_cat, dtype=torch.long),
                                  torch.tensor(y[tr_idx], dtype=torch.float32))
            va_ds = TensorDataset(torch.tensor(va_num_norm, dtype=torch.float32),
                                  torch.tensor(va_cat, dtype=torch.long),
                                  torch.tensor(y[va_idx], dtype=torch.float32))
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=8192, shuffle=False)

            net = TabMLP(len(NUM_FEATS), emb_dims, hidden_dims, dropout).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_W], device=device))

            best_state = None
            best_auc = -1
            bad = 0
            patience = 5
            for ep in range(NN_EPOCHS):
                net.train()
                for xb_n, xb_c, yb in tr_loader:
                    xb_n, xb_c, yb = xb_n.to(device), xb_c.to(device), yb.to(device)
                    opt.zero_grad()
                    loss_fn(net(xb_n, xb_c), yb).backward()
                    opt.step()
                net.eval()
                preds = []
                with torch.no_grad():
                    for xb_n, xb_c, _ in va_loader:
                        xb_n, xb_c = xb_n.to(device), xb_c.to(device)
                        preds.append(torch.sigmoid(net(xb_n, xb_c)).cpu().numpy())
                p = np.concatenate(preds)
                auc = roc_auc_score(y[va_idx], p)
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            net.load_state_dict(best_state)
            net.to(device).eval()

            # [FIX-3] 验证集分批推理
            va_num_t = torch.tensor(va_num_norm, dtype=torch.float32)
            va_cat_t = torch.tensor(va_cat, dtype=torch.long)
            p_va = predict_in_batches(net, va_num_t, va_cat_t, device)

            oof[va_idx] += p_va
            oof_count[va_idx] += 1                      # [FIX-2]

            # [FIX-3] 测试集分批推理
            te_num_t = torch.tensor(te_num_norm, dtype=torch.float32)
            te_cat_t = torch.tensor(te_cat, dtype=torch.long)
            p_te = predict_in_batches(net, te_num_t, te_cat_t, device)

            test_pred += p_te / n_runs
            aucs.append(best_auc)
            print(f"  seed{seed} fold{fold} | MLP AUC={best_auc:.5f}")

    oof /= np.maximum(oof_count, 1)                     # [FIX-2]
    return oof, test_pred, np.mean(aucs), np.std(aucs)


# 训练 CatBoost
print("\n=== 训练 CatBoost (5折) ===")
cb_oof, cb_test, cb_mean, cb_std = train_catboost_cv(best_cb_params, SEEDS)
print(f"CatBoost OOF AUC = {roc_auc_score(y, cb_oof):.5f} ± {cb_std:.5f}")

# 训练 MLP（残差版）
if USE_NN and best_nn_params is not None:
    print("\n=== 训练 PyTorch MLP（残差版）(5折) ===")
    nn_oof, nn_test, nn_mean, nn_std = train_mlp_cv(best_nn_params, SEEDS)
    print(f"MLP OOF AUC = {roc_auc_score(y, nn_oof):.5f} ± {nn_std:.5f}")
else:
    nn_oof = None

# ===================== 集成 =====================
if nn_oof is not None:
    auc_cb = roc_auc_score(y, cb_oof)
    auc_nn = roc_auc_score(y, nn_oof)
    w_cb = auc_cb / (auc_cb + auc_nn)
    w_nn = auc_nn / (auc_cb + auc_nn)
    print(f"\n集成权重: CatBoost={w_cb:.3f}, MLP={w_nn:.3f}")
    final_test = w_cb * cb_test + w_nn * nn_test
    final_oof = w_cb * cb_oof + w_nn * nn_oof
    print(f"集成后 OOF AUC = {roc_auc_score(y, final_oof):.5f}")
else:
    final_test = cb_test
    final_oof = cb_oof
    print("\n仅使用 CatBoost")

# ===================== 生成提交 =====================
# [FIX-4] 直接赋值，避免 merge 可能引入 NaN
submission = subm.copy()
submission[TARGET] = final_test
submission.to_csv("submission.csv", index=False)
print("\n提交文件已保存: submission.csv")