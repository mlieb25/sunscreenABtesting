"""
Sunscreen A/B Test Analysis
===========================
IV:  Brand (Ironcoast = 1, Banana Boat = 0)
DV:  Purchase likelihood (1–5 Likert)
Moderators: Gender (Male = 1), Age group (with 21-30 focal)
Hypothesis: Ironcoast ↑ purchase likelihood in males, especially 21-30
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

# ====================================================================
# 1. LOAD & CLEAN
# ====================================================================
df = pd.read_csv("BAMA580A+A_B+Test+-+Sunscreen_labels.csv", skiprows=[1, 2])
if 'Finished' in df.columns:
    df = df[df['Finished'] == True].copy()

df = df[['likelihood', 'gender', 'age', 'Stimuli_DO']].copy()
df = df.dropna()

likert_map = {
    'Extremely unlikely': 1,
    'Somewhat unlikely': 2,
    'Neither likely nor unlikely': 3,
    'Somewhat likely': 4,
    'Extremely likely': 5,
}
df['likelihood_num'] = df['likelihood'].map(likert_map)
df = df.dropna(subset=['likelihood_num'])

# Independent variable: brand
df['ironcoast'] = df['Stimuli_DO'].str.lower().str.contains('ironcoast').astype(int)
df['brand'] = df['ironcoast'].map({1: 'Ironcoast', 0: 'Banana Boat'})

# Moderators
df['male'] = (df['gender'] == 'Male').astype(int)
age_order = ['21-30', '31-40', '41-50', '50+']
df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)
df['age_21_30'] = (df['age'] == '21-30').astype(int)

N = len(df)
print(f"Sample: N = {N}  ({df['male'].sum()} male, {(1 - df['male']).sum():.0f} female)")
print(f"         {df['ironcoast'].sum()} Ironcoast, {(1 - df['ironcoast']).sum():.0f} Banana Boat")
print()

# ====================================================================
# 2. DESCRIPTIVE STATISTICS
# ====================================================================
print("=" * 70)
print("  DESCRIPTIVE STATISTICS")
print("=" * 70)

# Full cross-tab: gender × brand × age
desc = (df.groupby(['gender', 'brand', 'age'])['likelihood_num']
        .agg(['mean', 'std', 'count'])
        .round(3))
print(desc.to_string())
print()

# Marginal means: brand × gender
marg_gb = df.groupby(['brand', 'gender'])['likelihood_num'].agg(['mean', 'std', 'count']).round(3)
print("Brand × Gender marginal means:")
print(marg_gb.to_string())
print()

# ====================================================================
# 3. REGRESSION MODELS
# ====================================================================

# --- Model 1: Main effects only ---
mod1 = smf.ols('likelihood_num ~ ironcoast + male', data=df).fit()

# --- Model 2: Brand × Gender interaction ---
mod2 = smf.ols('likelihood_num ~ ironcoast * male', data=df).fit()

# --- Model 3: Brand × Gender × Age(21-30) — full moderation ---
mod3 = smf.ols('likelihood_num ~ ironcoast * male * age_21_30', data=df).fit()

print("=" * 70)
print("  MODEL 1: Main Effects  (likelihood ~ brand + gender)")
print("=" * 70)
print(mod1.summary2().tables[1].to_string())
print(f"\n  R² = {mod1.rsquared:.4f},  Adj R² = {mod1.rsquared_adj:.4f},  F = {mod1.fvalue:.3f},  p(F) = {mod1.f_pvalue:.4f}")

print()
print("=" * 70)
print("  MODEL 2: Gender Moderation  (likelihood ~ brand × gender)")
print("=" * 70)
print(mod2.summary2().tables[1].to_string())
print(f"\n  R² = {mod2.rsquared:.4f},  Adj R² = {mod2.rsquared_adj:.4f},  F = {mod2.fvalue:.3f},  p(F) = {mod2.f_pvalue:.4f}")

# F-test: Model 2 vs Model 1
anova_12 = sm.stats.anova_lm(mod1, mod2)
print(f"\n  ΔR² (Model 2 vs 1) = {mod2.rsquared - mod1.rsquared:.4f},  F-change = {anova_12.iloc[1]['F']:.3f},  p = {anova_12.iloc[1]['Pr(>F)']:.4f}")

print()
print("=" * 70)
print("  MODEL 3: Full Moderation  (likelihood ~ brand × gender × age_21_30)")
print("=" * 70)
print(mod3.summary2().tables[1].to_string())
print(f"\n  R² = {mod3.rsquared:.4f},  Adj R² = {mod3.rsquared_adj:.4f},  F = {mod3.fvalue:.3f},  p(F) = {mod3.f_pvalue:.4f}")

# F-test: Model 3 vs Model 2
anova_23 = sm.stats.anova_lm(mod2, mod3)
print(f"\n  ΔR² (Model 3 vs 2) = {mod3.rsquared - mod2.rsquared:.4f},  F-change = {anova_23.iloc[1]['F']:.3f},  p = {anova_23.iloc[1]['Pr(>F)']:.4f}")

# Interpret the three-way interaction
coef_3way = mod3.params.get('ironcoast:male:age_21_30', 0)
p_3way    = mod3.pvalues.get('ironcoast:male:age_21_30', 1)
print(f"\n  Three-way interaction (brand × male × age_21_30):")
print(f"  β = {coef_3way:.3f},  p = {p_3way:.4f}")

# ====================================================================
# 4. FOCAL SIMPLE-EFFECTS t-TESTS
# ====================================================================
print()
print("=" * 70)
print("  SIMPLE EFFECTS: Ironcoast vs. Banana Boat within subgroups")
print("=" * 70)

def run_ttest(sub_df, label):
    ir = sub_df[sub_df['ironcoast'] == 1]['likelihood_num']
    bb = sub_df[sub_df['ironcoast'] == 0]['likelihood_num']
    if len(ir) < 2 or len(bb) < 2:
        print(f"  {label:42s}  <2 obs per group — skipped")
        return None
    t = stats.ttest_ind(ir, bb, equal_var=False)
    n1, n2 = len(ir), len(bb)
    s1, s2 = ir.std(ddof=1), bb.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d = (ir.mean() - bb.mean()) / sp
    sig = "*" if t.pvalue < 0.05 else ""
    print(f"  {label:42s}  Δ={ir.mean()-bb.mean():+.3f}  d={d:+.3f}  t={t.statistic:+.3f}  p={t.pvalue:.4f}{sig}  (n={n1},{n2})")
    return {'label': label, 'ir_mean': ir.mean(), 'bb_mean': bb.mean(),
            'diff': ir.mean()-bb.mean(), 'd': d, 't': t.statistic,
            'p': t.pvalue, 'n_ir': n1, 'n_bb': n2}

results = []
for g_name, g_val in [('Male', 1), ('Female', 0)]:
    r = run_ttest(df[df['male'] == g_val], f"{g_name} (all ages)")
    if r: results.append(r)
    for age_grp in age_order:
        sub = df[(df['male'] == g_val) & (df['age'] == age_grp)]
        r = run_ttest(sub, f"  → {g_name}, {age_grp}")
        if r: results.append(r)
    print()

print("(* = p < .05 two-tailed)")

# ====================================================================
# 5. POWER ANALYSIS
# ====================================================================
print()
print("=" * 70)
print("  POWER ANALYSIS")
print("=" * 70)

power_calc = TTestIndPower()

# a) Post-hoc: overall brand effect (full sample)
ir_all = df[df['ironcoast'] == 1]['likelihood_num']
bb_all = df[df['ironcoast'] == 0]['likelihood_num']
d_all = ((ir_all.mean() - bb_all.mean()) /
         np.sqrt(((len(ir_all)-1)*ir_all.std(ddof=1)**2 + (len(bb_all)-1)*bb_all.std(ddof=1)**2) /
                 (len(ir_all)+len(bb_all)-2)))

pow_all = power_calc.solve_power(abs(d_all), nobs1=len(ir_all),
                                  ratio=len(bb_all)/len(ir_all), alpha=0.05)
req_all = power_calc.solve_power(abs(d_all), power=0.80, ratio=1.0, alpha=0.05)

print(f"\n  A. Full sample (brand main effect)")
print(f"     d = {d_all:.3f},  n = {len(ir_all)},{len(bb_all)}")
print(f"     Post-hoc power: {pow_all:.3f}")
print(f"     Required n/group for 80% power: {int(np.ceil(req_all)):,}")
print(f"     Required total survey respondents: {int(np.ceil(req_all))*2:,}")

# b) Post-hoc: males overall
ir_m = df[(df['ironcoast']==1)&(df['male']==1)]['likelihood_num']
bb_m = df[(df['ironcoast']==0)&(df['male']==1)]['likelihood_num']
d_m = ((ir_m.mean() - bb_m.mean()) /
       np.sqrt(((len(ir_m)-1)*ir_m.std(ddof=1)**2 + (len(bb_m)-1)*bb_m.std(ddof=1)**2) /
               (len(ir_m)+len(bb_m)-2)))
pow_m = power_calc.solve_power(abs(d_m), nobs1=len(ir_m),
                                ratio=len(bb_m)/len(ir_m), alpha=0.05)
req_m = power_calc.solve_power(abs(d_m) if abs(d_m) > 0.01 else 0.01,
                                power=0.80, ratio=1.0, alpha=0.05)
pct_male = df['male'].mean()

print(f"\n  B. Males overall (brand effect among men)")
print(f"     d = {d_m:.3f},  n = {len(ir_m)},{len(bb_m)}")
print(f"     Post-hoc power: {pow_m:.3f}")
print(f"     Required n/group (males): {int(np.ceil(req_m)):,}")
print(f"     Required total survey respondents (at {pct_male:.0%} male): {int(np.ceil(int(np.ceil(req_m))*2 / pct_male)):,}")

# c) Post-hoc: males 21-30 (focal hypothesis)
ir_m21 = df[(df['ironcoast']==1)&(df['male']==1)&(df['age']=='21-30')]['likelihood_num']
bb_m21 = df[(df['ironcoast']==0)&(df['male']==1)&(df['age']=='21-30')]['likelihood_num']
d_m21 = ((ir_m21.mean() - bb_m21.mean()) /
         np.sqrt(((len(ir_m21)-1)*ir_m21.std(ddof=1)**2 + (len(bb_m21)-1)*bb_m21.std(ddof=1)**2) /
                 (len(ir_m21)+len(bb_m21)-2)))
pow_m21 = power_calc.solve_power(abs(d_m21), nobs1=len(ir_m21),
                                  ratio=len(bb_m21)/len(ir_m21), alpha=0.05)
req_m21 = power_calc.solve_power(abs(d_m21), power=0.80, ratio=1.0, alpha=0.05)
pct_m21 = ((df['male']==1)&(df['age']=='21-30')).mean()

print(f"\n  C. Males 21-30 (focal subgroup)")
print(f"     d = {d_m21:.3f},  n = {len(ir_m21)},{len(bb_m21)}")
print(f"     Post-hoc power: {pow_m21:.3f}")
print(f"     Required n/group (male 21-30): {int(np.ceil(req_m21))}")
print(f"     Required total survey respondents (at {pct_m21:.0%} male 21-30): {int(np.ceil(int(np.ceil(req_m21))*2 / pct_m21))}")

# d) Post-hoc power for the regression model (Model 3 R²)
f2 = mod3.rsquared / (1 - mod3.rsquared)
df_num = mod3.df_model
df_den = mod3.df_resid
from statsmodels.stats.power import FTestPower
fpow = FTestPower()
pow_mod3 = fpow.solve_power(effect_size=np.sqrt(f2), df_num=df_num, df_denom=df_den, alpha=0.05)

print(f"\n  D. Regression Model 3 (full moderation)")
print(f"     R² = {mod3.rsquared:.4f},  f² = {f2:.4f}")
print(f"     Post-hoc power (F-test): {pow_mod3:.3f}")

# Required N for 80% power on Model 3's R²
# Iterate to find the total N where power ≥ 0.80
req_total_mod3 = None
for n_try in range(int(df_num) + 3, 5000):
    dd = n_try - int(df_num) - 1
    if dd <= 0:
        continue
    p_try = fpow.solve_power(effect_size=np.sqrt(f2), df_num=df_num, df_denom=dd, alpha=0.05)
    if p_try >= 0.80:
        req_total_mod3 = n_try
        break
if req_total_mod3:
    print(f"     Required total N for 80% power: {req_total_mod3}")
else:
    print(f"     Required total N for 80% power: >5,000")

# ====================================================================
# 6. VISUALIZATIONS
# ====================================================================

# Color palette
IRON  = '#0D6E8A'
BB    = '#E87830'
SIG   = '#27AE60'
NSIG  = '#E74C3C'
BG    = '#F5F6FA'
GRID  = '#D9DCE6'
TXT   = '#1E2340'
MALE_BG  = '#E3F0FA'
FEM_BG   = '#FDEAE5'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor': BG,
    'axes.edgecolor': GRID,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'text.color': TXT,
    'axes.labelcolor': TXT,
    'xtick.color': TXT,
    'ytick.color': TXT,
})

fig = plt.figure(figsize=(20, 20))
gs = GridSpec(4, 4, figure=fig, hspace=0.42, wspace=0.38,
             left=0.06, right=0.97, top=0.93, bottom=0.04)

fig.suptitle('Sunscreen A/B Test — Ironcoast vs. Banana Boat\n'
             'Purchase Likelihood with Gender × Age Moderation',
             fontsize=16, fontweight='bold', color=TXT, y=0.97)

# ---- (A) Mean ± 95% CI by gender × brand ----
ax_a = fig.add_subplot(gs[0, :2])
groups = [
    ('Male\nIroncoast',  ir_m,  IRON, MALE_BG),
    ('Male\nBanana Boat', bb_m, BB,   MALE_BG),
    ('Female\nIroncoast',
     df[(df['ironcoast']==1)&(df['male']==0)]['likelihood_num'], IRON, FEM_BG),
    ('Female\nBanana Boat',
     df[(df['ironcoast']==0)&(df['male']==0)]['likelihood_num'], BB,   FEM_BG),
]
x_pos = np.arange(len(groups))
for i, (lbl, data, col, bgcol) in enumerate(groups):
    m = data.mean()
    ci = stats.t.ppf(0.975, len(data)-1) * data.sem()
    bar = ax_a.bar(i, m, color=col, width=0.55, edgecolor='white', linewidth=1.5, zorder=3)
    ax_a.errorbar(i, m, yerr=ci, fmt='none', color='#333', capsize=6, lw=2, zorder=4)
    ax_a.text(i, m + ci + 0.12, f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')
    # Background shading
    if i in [0,1]:
        ax_a.axvspan(i-0.4, i+0.4, color=bgcol, alpha=0.25, zorder=0)
    else:
        ax_a.axvspan(i-0.4, i+0.4, color=bgcol, alpha=0.25, zorder=0)
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([g[0] for g in groups], fontsize=9)
ax_a.set_ylim(1, 5.5)
ax_a.set_ylabel('Mean Purchase Likelihood (1–5)', fontsize=9)
ax_a.set_title('A  Mean Purchase Likelihood by Gender × Brand\n(with 95% CI)',
               fontsize=11, fontweight='bold', pad=8)
iron_p = mpatches.Patch(color=IRON, label='Ironcoast')
bb_p   = mpatches.Patch(color=BB, label='Banana Boat')
ax_a.legend(handles=[iron_p, bb_p], fontsize=9, loc='upper right')

# ---- (B) Interaction plot: brand × gender ----
ax_b = fig.add_subplot(gs[0, 2:])
for gen, style, col in [('Male', 'o-', '#2980B9'), ('Female', 's--', '#C0392B')]:
    means = [df[(df['brand']==b)&(df['gender']==gen)]['likelihood_num'].mean()
             for b in ['Banana Boat', 'Ironcoast']]
    ax_b.plot(['Banana Boat', 'Ironcoast'], means, style, color=col, lw=2.5,
              ms=10, label=gen, zorder=3)
    for xi, m in enumerate(means):
        ax_b.text(xi, m + 0.12, f'{m:.2f}', ha='center', fontsize=9.5,
                  fontweight='bold', color=col)
ax_b.set_ylim(1, 5.2)
ax_b.set_ylabel('Mean Purchase Likelihood', fontsize=9)
ax_b.set_xlabel('Brand', fontsize=9)
ax_b.set_title('B  Interaction Plot: Brand × Gender',
               fontsize=11, fontweight='bold', pad=8)
ax_b.legend(fontsize=10)

# ---- (C) Brand × Age × Gender grouped bar ----
ax_c = fig.add_subplot(gs[1, :])
x = np.arange(len(age_order))
w = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]
combos = [
    ('Male – Ironcoast',  1, 1, IRON, '///'),
    ('Male – Banana Boat', 1, 0, BB,  '///'),
    ('Female – Ironcoast', 0, 1, IRON, ''),
    ('Female – Banana Boat', 0, 0, BB, ''),
]
for idx, (lbl, gen, brand, col, hatch) in enumerate(combos):
    means = []
    for age in age_order:
        sub = df[(df['male']==gen)&(df['ironcoast']==brand)&(df['age']==age)]['likelihood_num']
        means.append(sub.mean() if len(sub) > 0 else np.nan)
    bars = ax_c.bar(x + offsets[idx]*w, means, w*0.9, label=lbl, color=col,
                    alpha=0.85 if hatch else 0.5, hatch=hatch,
                    edgecolor='white', linewidth=1, zorder=3)
    for xi, m in enumerate(means):
        if not np.isnan(m):
            ax_c.text(x[xi] + offsets[idx]*w, m + 0.08, f'{m:.1f}',
                      ha='center', fontsize=7.5, fontweight='bold')

# Star the focal subgroup
focal_ir  = df[(df['male']==1)&(df['ironcoast']==1)&(df['age']=='21-30')]['likelihood_num'].mean()
ax_c.annotate('★ Focal\nhypothesis', xy=(0 + offsets[0]*w, focal_ir + 0.2),
              xytext=(0 + offsets[0]*w - 0.15, focal_ir + 0.8),
              fontsize=8.5, fontweight='bold', color=SIG,
              arrowprops=dict(arrowstyle='->', color=SIG, lw=1.5),
              ha='center')
ax_c.set_xticks(x)
ax_c.set_xticklabels(age_order, fontsize=10)
ax_c.set_ylim(1, 6)
ax_c.set_ylabel('Mean Purchase Likelihood', fontsize=9)
ax_c.set_xlabel('Age Group', fontsize=9)
ax_c.set_title('C  Purchase Likelihood by Brand × Gender × Age Group\n(hatched = male, solid = female)',
               fontsize=11, fontweight='bold', pad=8)
ax_c.legend(fontsize=8, ncol=4, loc='upper right')

# ---- (D) Effect size (Cohen's d) by subgroup — horizontal lollipop ----
ax_d = fig.add_subplot(gs[2, :2])
res_df = pd.DataFrame(results)
res_df = res_df.iloc[::-1]  # reverse for nice ordering
y_d = np.arange(len(res_df))
colors_d = [SIG if p < 0.05 else NSIG for p in res_df['p']]
ax_d.hlines(y_d, 0, res_df['d'], color=colors_d, linewidth=2.5, zorder=3)
ax_d.scatter(res_df['d'], y_d, color=colors_d, s=80, zorder=4, edgecolor='white', linewidth=1.5)
ax_d.axvline(0, color=TXT, linewidth=1.2)
for thresh, lbl in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
    ax_d.axvline(thresh, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax_d.axvline(-thresh, color='gray', ls='--', lw=0.7, alpha=0.5)
    ax_d.text(thresh, len(res_df)-0.3, lbl, fontsize=7, ha='center', color='gray')
for i, row in enumerate(res_df.itertuples()):
    p_str = f"p={row.p:.3f}"
    ax_d.text(row.d + (0.06 if row.d >= 0 else -0.06), i,
              f"d={row.d:+.2f}  {p_str}", va='center',
              ha='left' if row.d >= 0 else 'right', fontsize=8)
ax_d.set_yticks(y_d)
ax_d.set_yticklabels(res_df['label'], fontsize=9)
ax_d.set_xlabel("Cohen's d  (Ironcoast − Banana Boat)", fontsize=9)
ax_d.set_title("D  Effect Sizes by Subgroup\n(green = p<.05, red = p≥.05)",
               fontsize=11, fontweight='bold', pad=8)
sig_p  = mpatches.Patch(color=SIG, label='p < .05')
ns_p   = mpatches.Patch(color=NSIG, label='p ≥ .05')
ax_d.legend(handles=[sig_p, ns_p], fontsize=8, loc='lower right')

# ---- (E) Likert distribution — focal subgroup ----
ax_e = fig.add_subplot(gs[2, 2:])
likert_labels = ['Ext. Unlikely', 'Somewhat\nUnlikely', 'Neutral', 'Somewhat\nLikely', 'Ext. Likely']
likert_colors = ['#C0392B', '#E67E22', '#95A5A6', '#27AE60', '#1A5276']

grps = [
    (f'Ironcoast\nMale 21-30 (n={len(ir_m21)})', ir_m21),
    (f'Banana Boat\nMale 21-30 (n={len(bb_m21)})', bb_m21),
    (f'Ironcoast\nAll others (n={len(ir_all)-len(ir_m21)})',
     df[(df['ironcoast']==1)&~((df['male']==1)&(df['age']=='21-30'))]['likelihood_num']),
    (f'Banana Boat\nAll others (n={len(bb_all)-len(bb_m21)})',
     df[(df['ironcoast']==0)&~((df['male']==1)&(df['age']=='21-30'))]['likelihood_num']),
]
y_e = np.arange(len(grps))
left = np.zeros(len(grps))
for li, (lval, lcol) in enumerate(zip([1,2,3,4,5], likert_colors)):
    widths = [((g == lval).sum() / len(g) * 100) if len(g) > 0 else 0 for _, g in grps]
    bars = ax_e.barh(y_e, widths, left=left, color=lcol, height=0.55,
                     edgecolor='white', linewidth=0.8, label=likert_labels[li])
    for bar, wv in zip(bars, widths):
        if wv > 8:
            ax_e.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                     f'{wv:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    left += widths
ax_e.set_yticks(y_e)
ax_e.set_yticklabels([g[0] for g in grps], fontsize=8.5)
ax_e.set_xlim(0, 100)
ax_e.set_xlabel('% of respondents', fontsize=9)
ax_e.set_title('E  Response Distribution\n(focal subgroup vs. others)',
               fontsize=11, fontweight='bold', pad=8)
ax_e.legend(fontsize=7.5, ncol=5, loc='lower right', framealpha=0.9)
ax_e.grid(axis='y', alpha=0)

# ---- (F) Power analysis summary visual ----
ax_f = fig.add_subplot(gs[3, :])
ax_f.set_xlim(0, 10)
ax_f.set_ylim(0, 4)
ax_f.axis('off')
ax_f.set_facecolor('white')
ax_f.set_title('F  Power Analysis Summary',
               fontsize=11, fontweight='bold', pad=8)

power_data = [
    ("Full Sample\n(brand main effect)", d_all, pow_all, int(np.ceil(req_all))*2, N),
    ("Males Overall\n(brand among men)", d_m, pow_m, int(np.ceil(int(np.ceil(req_m))*2/pct_male)), len(ir_m)+len(bb_m)),
    ("Males 21-30\n(focal hypothesis)", d_m21, pow_m21, int(np.ceil(int(np.ceil(req_m21))*2/pct_m21)), len(ir_m21)+len(bb_m21)),
]

for i, (lbl, d_val, pow_val, req_total, n_curr) in enumerate(power_data):
    cx = 1.7 + i * 3.0
    cy = 2.2

    # Box
    box_col = SIG if pow_val >= 0.80 else ('#F39C12' if pow_val >= 0.50 else NSIG)
    rect = mpatches.FancyBboxPatch((cx - 1.2, cy - 1.6), 2.4, 3.2,
                                    boxstyle="round,pad=0.12",
                                    facecolor=box_col, alpha=0.08,
                                    edgecolor=box_col, linewidth=2)
    ax_f.add_patch(rect)

    ax_f.text(cx, cy + 1.3, lbl, ha='center', va='center', fontsize=9.5, fontweight='bold')
    ax_f.text(cx, cy + 0.5, f"|d| = {abs(d_val):.3f}", ha='center', fontsize=9)
    ax_f.text(cx, cy + 0.0, f"Current n = {n_curr}", ha='center', fontsize=9)

    # Power meter
    pow_pct = pow_val * 100
    ax_f.text(cx, cy - 0.5, f"Power: {pow_pct:.1f}%", ha='center', fontsize=11,
              fontweight='bold', color=box_col)

    # Target bar
    bar_w = 1.8
    bar_h = 0.2
    bx = cx - bar_w / 2
    by = cy - 1.0
    ax_f.add_patch(mpatches.Rectangle((bx, by), bar_w, bar_h, fc='#ddd', ec='#aaa', zorder=2))
    fill_w = min(bar_w * pow_val, bar_w)
    ax_f.add_patch(mpatches.Rectangle((bx, by), fill_w, bar_h, fc=box_col, alpha=0.6, zorder=3))
    # 80% target line
    ax_f.plot([bx + bar_w*0.8, bx + bar_w*0.8], [by - 0.05, by + bar_h + 0.05],
              color='black', linewidth=1.5, zorder=4)
    ax_f.text(bx + bar_w*0.8, by - 0.12, '80%', ha='center', fontsize=7, color='#555')

    ax_f.text(cx, cy - 1.35, f"Need N ≈ {req_total:,} for 80%", ha='center',
              fontsize=8.5, color='#555', style='italic')

out_path = 'output/full_analysis_dashboard.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n📊 Dashboard saved → {out_path}")

# Save all simple-effects to CSV
pd.DataFrame(results).to_csv('output/simple_effects.csv', index=False)
print("📄 CSV saved → output/simple_effects.csv")