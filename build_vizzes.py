"""Build final FOR/AGAINST composite images + 3-page checkpoint PDF for DSC106 Project 2."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cmx
import matplotlib.colors as mcol
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

df = pd.read_csv('data.csv')

RATE_RES = 'No. of abortions per 1,000 women aged 15–44, by state of residence, 2020'
RATE_OCC = 'No. of abortions per 1,000 women aged 15–44, by state of occurrence, 2020'
TRAVEL = '% of residents obtaining abortions who traveled out of state for care, 2020'
NO_CLINIC = '% of counties without a known clinic, 2020'
CHG_RATE = '% change in abortion rate, 2017-2020'

for c in [RATE_RES, RATE_OCC, TRAVEL, NO_CLINIC, CHG_RATE]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
    'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}
df['abbr'] = df['U.S. State'].map(STATE_ABBR)

TILE = {
    'AK':(0,0),                                                                               'ME':(10,0),
    'VT':(9,1),'NH':(10,1),
    'WA':(1,2),'ID':(2,2),'MT':(3,2),'ND':(4,2),'MN':(5,2),'IL':(6,2),'WI':(7,2),'MI':(8,2),'NY':(9,2),'MA':(10,2),
    'OR':(1,3),'NV':(2,3),'WY':(3,3),'SD':(4,3),'IA':(5,3),'IN':(6,3),'OH':(7,3),'PA':(8,3),'NJ':(9,3),'CT':(10,3),'RI':(11,3),
    'CA':(1,4),'UT':(2,4),'CO':(3,4),'NE':(4,4),'MO':(5,4),'KY':(6,4),'WV':(7,4),'VA':(8,4),'MD':(9,4),'DE':(10,4),
                'AZ':(2,5),'NM':(3,5),'KS':(4,5),'AR':(5,5),'TN':(6,5),'NC':(7,5),'SC':(8,5),'DC':(9,5),
    'HI':(0,6),                      'OK':(4,6),'LA':(5,6),'MS':(6,6),'AL':(7,6),'GA':(8,6),
                                     'TX':(4,7),                                   'FL':(9,7),
}

# Pre-Dobbs "restrictive" states (trigger laws + hostile TRAP regimes circa 2020, per Guttmacher)
RESTRICTIVE = {'AL','AR','GA','ID','IN','KY','LA','MS','MO','ND','OK','SD','TN','TX','UT','WV','WI','WY'}

# Categorical palette: red for restrictive states, gray for others
RED  = '#C8102E'   # restrictive states
GRAY = '#7C7E80'   # other states

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def tile_map(ax, values, cmap_restrictive='Reds', cmap_other='Greys',
             vmin=None, vmax=None, value_fmt='{:.1f}'):
    """Tile-grid choropleth. Restrictive states use red ramp + red outline,
    other states use gray ramp. Same encoding as the bars next to the map."""
    cm_r = plt.get_cmap(cmap_restrictive)
    cm_o = plt.get_cmap(cmap_other)
    vmin = vmin if vmin is not None else np.nanmin(values.values)
    vmax = vmax if vmax is not None else np.nanmax(values.values)
    for _, row in df.iterrows():
        ab = row['abbr']
        if ab not in TILE:
            continue
        c, r = TILE[ab]
        v = values.get(row.name, np.nan)
        is_restr = ab in RESTRICTIVE
        if pd.isna(v):
            color = '#eeeeee'
            t = 0
        else:
            t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            t = max(0, min(1, t))
            cm = cm_r if is_restr else cm_o
            color = cm(0.18 + t * 0.65)
        edge = RED if is_restr else 'white'
        lw = 1.8 if is_restr else 0.6
        rect = mpatches.FancyBboxPatch(
            (c, -r), 0.92, 0.92,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=lw, edgecolor=edge, facecolor=color
        )
        ax.add_patch(rect)
        txt_color = 'white' if t > 0.55 else '#222'
        ax.text(c + 0.46, -r + 0.58, ab, ha='center', va='center',
                fontsize=9, fontweight='bold', color=txt_color)
        if not pd.isna(v):
            ax.text(c + 0.46, -r + 0.28, value_fmt.format(v), ha='center', va='center',
                    fontsize=7.5, color=txt_color)
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-8, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def add_dual_legend(fig, rect, vmin, vmax, label, fmt='{:.0f}'):
    """Two stacked horizontal color ramps (red, gray) with shared scale."""
    cax_r = fig.add_axes([rect[0], rect[1] + rect[3] * 0.55, rect[2], rect[3] * 0.4])
    cax_o = fig.add_axes([rect[0], rect[1], rect[2], rect[3] * 0.4])
    sm_r = cmx.ScalarMappable(cmap='Reds',  norm=mcol.Normalize(vmin=vmin, vmax=vmax))
    sm_o = cmx.ScalarMappable(cmap='Greys', norm=mcol.Normalize(vmin=vmin, vmax=vmax))
    sm_r.set_array([]); sm_o.set_array([])
    cb_r = plt.colorbar(sm_r, cax=cax_r, orientation='horizontal')
    cb_o = plt.colorbar(sm_o, cax=cax_o, orientation='horizontal')
    cb_r.set_ticks([]); cb_o.ax.tick_params(labelsize=7.5)
    cb_o.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: fmt.format(x)))
    cax_r.text(-0.01, 0.5, 'Restrictive', transform=cax_r.transAxes,
               ha='right', va='center', fontsize=8, color=RED, fontweight='bold')
    cax_o.text(-0.01, 0.5, 'Other', transform=cax_o.transAxes,
               ha='right', va='center', fontsize=8, color=GRAY, fontweight='bold')
    fig.text(rect[0] + rect[2] / 2, rect[1] - 0.018, label,
             ha='center', va='top', fontsize=8.5, color='#444')


def add_fig_colorbar(fig, rect, cmap, vmin, vmax, label, fmt=None):
    """Add a horizontal colorbar at figure-level coordinates rect=[x,y,w,h]."""
    sm = cmx.ScalarMappable(cmap=cmap, norm=mcol.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes(rect)
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if fmt is not None:
        cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: fmt.format(x)))
    return cax


# ============================================================
# Reusable subplots
# ============================================================
def comparison_bars(ax, restrictive_val, other_val, ylabel, value_fmt='{:.1f}'):
    """Two-bar comparison: restrictive vs other states."""
    bars = ax.bar(['Restrictive\nstates', 'Other\nstates'],
                  [restrictive_val, other_val],
                  color=[RED, GRAY], width=0.55, edgecolor='white', linewidth=1.2)
    ax.bar_label(bars, labels=[value_fmt.format(restrictive_val), value_fmt.format(other_val)],
                 padding=4, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    lo = min(0, restrictive_val, other_val)
    hi = max(0, restrictive_val, other_val)
    pad = (hi - lo) * 0.20
    ax.set_ylim(lo - pad if lo < 0 else 0, hi + pad)
    if lo < 0:
        ax.axhline(0, color='#222', lw=0.8)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9)


def diverging_gap(ax):
    """Residence vs occurrence diverging bar chart."""
    sub = df.dropna(subset=[RATE_RES, RATE_OCC]).copy()
    sub = sub[sub['abbr'] != 'DC']
    sub['gap'] = sub[RATE_RES] - sub[RATE_OCC]
    exporters = sub.nlargest(8, 'gap')
    importers = sub.nsmallest(7, 'gap')
    picked = pd.concat([exporters, importers]).sort_values('gap')
    ypos = np.arange(len(picked))
    colors = [GRAY if g < 0 else RED for g in picked['gap'].values]
    ax.barh(ypos, picked['gap'].values, color=colors, edgecolor='white', linewidth=0.8)
    ax.axvline(0, color='#222', lw=1)
    for i, g in enumerate(picked['gap'].values):
        offset = 0.15 if g >= 0 else -0.15
        ha = 'left' if g >= 0 else 'right'
        ax.text(g + offset, i, f'{g:+.1f}', va='center', ha=ha, fontsize=8.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(picked['U.S. State'].values, fontsize=9)
    ax.set_xlabel('Rate by residence minus rate by occurrence (per 1,000 women)', fontsize=10)
    ax.grid(axis='x', alpha=0.25)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 0.8, xmax + 1.3)
    e = mpatches.Patch(color=RED,  label='Residents leave for care')
    i = mpatches.Patch(color=GRAY, label='Residents arrive from elsewhere')
    ax.legend(handles=[e, i], loc='lower right', frameon=False, fontsize=9)


def travel_bar_chart(ax, n=15):
    """Top n states by % of residents who traveled out of state for an abortion.
    Highest value (Missouri 99%) shown at the top of the chart."""
    sub = df.dropna(subset=[TRAVEL]).copy()
    sub = sub.sort_values(TRAVEL, ascending=False).head(n)
    colors = [RED if ab in RESTRICTIVE else GRAY for ab in sub['abbr']]
    ypos = np.arange(len(sub))
    ax.barh(ypos, sub[TRAVEL].values, color=colors, edgecolor='white', linewidth=0.8)
    for i, v in enumerate(sub[TRAVEL].values):
        ax.text(v + 1.5, i, f'{int(v)}%', va='center', fontsize=9, color='#333')
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub['U.S. State'].values, fontsize=9)
    ax.set_xlabel('% of residents who crossed state lines for an abortion (2020)', fontsize=10)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.25)
    r = mpatches.Patch(color=RED,  label='Restrictive-law states')
    o = mpatches.Patch(color=GRAY, label='Other states')
    ax.legend(handles=[r, o], loc='lower right', frameon=False, fontsize=9)


# ============================================================
# Final FOR composite (Side A)
# ============================================================
def build_for_image(out='for.png'):
    fig = plt.figure(figsize=(14, 12.5))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.975, 'SIDE A: supporting the proposition',
             ha='center', fontsize=11, color=RED, fontweight='bold')

    fig.text(0.04, 0.948, 'Restrictive States Have the Lowest Abortion Rates',
             fontsize=15.5, fontweight='bold')
    fig.text(0.04, 0.928,
             'Abortions per 1,000 women aged 15-44 by state of residence (2020). '
             'Utah, South Dakota, Idaho, and Wyoming all sit below 6.5; California is above 24.',
             fontsize=10, color='#555')

    # Map (left)
    ax_map = fig.add_axes([0.0, 0.56, 0.58, 0.34])
    tile_map(ax_map, df[RATE_RES], vmin=4, vmax=30, value_fmt='{:.1f}')
    add_dual_legend(fig, [0.13, 0.525, 0.30, 0.020], 4, 30,
                    'Abortions per 1,000 women aged 15-44',
                    fmt='{:.0f}')

    # Right-side aggregates: averages and 2017->2020 change
    restr_avg = df[df['abbr'].isin(RESTRICTIVE)][RATE_RES].mean()
    other_avg = df[~df['abbr'].isin(RESTRICTIVE) & df['abbr'].notna() & (df['abbr'] != 'DC')][RATE_RES].mean()
    ax_a = fig.add_axes([0.62, 0.62, 0.16, 0.22])
    comparison_bars(ax_a, restr_avg, other_avg, 'Per 1,000 women aged 15-44')
    ax_a.set_title(
        'Average abortion rate per 1,000 women 15-44\n'
        f'is {other_avg/restr_avg:.1f}× higher in other states '
        f'({restr_avg:.1f} vs {other_avg:.1f})',
        fontsize=9.5, pad=6)

    restr_chg = df[df['abbr'].isin(RESTRICTIVE)][CHG_RATE].mean()
    other_chg = df[~df['abbr'].isin(RESTRICTIVE) & df['abbr'].notna() & (df['abbr'] != 'DC')][CHG_RATE].mean()
    ax_b = fig.add_axes([0.81, 0.62, 0.16, 0.22])
    comparison_bars(ax_b, restr_chg, other_chg, '% change, 2017-2020',
                    value_fmt='{:+.1f}%')
    ax_b.set_title(
        'Average % change in abortion rate\n'
        'since 2017, when many state TRAP\n'
        'and trigger laws tightened',
        fontsize=9.5, pad=6)

    # Connector lines: map's restrictive cluster -> red bar in each chart.
    # The line literally points from the red-outlined states on the map
    # to the red bar that summarizes them.
    import matplotlib.lines as mlines
    map_anchor = (0.45, 0.70)   # right edge of restrictive cluster on map
    for bar_x in (0.648, 0.838):  # x of the red 'Restrictive states' bar in each chart
        line = mlines.Line2D(
            [map_anchor[0], bar_x], [map_anchor[1], 0.738],
            transform=fig.transFigure,
            color=RED, lw=1.0, alpha=0.45, linestyle=(0, (4, 3)),
            zorder=0
        )
        fig.add_artist(line)
    fig.text(0.555, 0.755, 'averaging the\nred-outlined\nstates →',
             fontsize=8, color=RED, alpha=0.8, ha='center', va='center',
             style='italic')

    # Bottom: scatter (taller chart + smaller dots + wider jitter to reduce overlap)
    fig.text(0.04, 0.475, 'Where there are no clinics, there are fewer abortions',
             fontsize=13, fontweight='bold')
    fig.text(0.04, 0.456,
             'Each dot is a state. As the share of counties without an abortion clinic rises, the abortion rate falls.',
             fontsize=9.5, color='#555')
    ax_sc = fig.add_axes([0.07, 0.06, 0.88, 0.38])
    sub = df.dropna(subset=[NO_CLINIC, RATE_RES]).copy()
    sub = sub[sub['abbr'] != 'DC'].reset_index(drop=True)
    x_raw = sub[NO_CLINIC].values
    y_raw = sub[RATE_RES].values
    rng = np.random.default_rng(7)
    x_j = x_raw + rng.uniform(-3.5, 3.5, len(x_raw))
    y_j = y_raw + rng.uniform(-0.55, 0.55, len(y_raw))
    colors = [RED if ab in RESTRICTIVE else GRAY for ab in sub['abbr']]
    ax_sc.scatter(x_j, y_j, s=28, c=colors, edgecolors='white', linewidths=0.7,
                  zorder=3, alpha=0.85)
    m, b = np.polyfit(x_raw, y_raw, 1)
    xs = np.linspace(0, 100, 50)
    ax_sc.plot(xs, m*xs + b, '--', color='#222', lw=1.3, zorder=2, alpha=0.7)
    # 18 restrictive states are stacked at x=93-99, so the right-cluster labels
    # are anchored to fixed positions outside the cluster with thin leader lines.
    # Two columns: right column at x≈112, left column tucked just inside the cluster.
    cluster_targets = {
        # state -> (target_x, target_y) in data coords; lines connect dot -> label
        'GA':(108, 18.0),   # y=16.6
        'OK':(108, 13.0),   # y=10.7
        'TX':(82,  13.5),   # y=10.1, label up-left
        'MS':(112, 10.5),   # y=9.9
        'MO':(85,  11.0),   # y=9.9, label up-left (away from MS)
        'AL':(82,  9.5),    # y=9.5, label left
        'LA':(110, 8.5),    # y=8.4
        'IN':(82,  8.0),    # y=8.2
        'AR':(112, 7.0),    # y=7.8
        'TN':(82,  6.5),    # y=7.8
        'WI':(108, 5.5),    # y=7.6
        'KY':(112, 4.5),    # y=6.7
        'WY':(82,  5.0),    # y=6.4
        'ND':(108, 3.0),    # y=6.4
        'ID':(82,  3.5),    # y=6.0
        'WV':(108, 1.5),    # y=5.7
        'UT':(82,  2.0),    # y=4.5
        'SD':(108, 0.5),    # y=4.1
    }
    for i, row in sub.iterrows():
        ab = row['abbr']
        if ab in cluster_targets:
            tx, ty = cluster_targets[ab]
            ax_sc.annotate(ab, (x_j[i], y_j[i]),
                           xytext=(tx, ty), textcoords='data',
                           fontsize=8.5, color='#222', ha='center', va='center',
                           arrowprops=dict(arrowstyle='-', color='#888',
                                           lw=0.5, alpha=0.55,
                                           shrinkA=2, shrinkB=4))
    # Permissive / non-cluster labels stay as simple offsets
    permissive_offsets = {
        'NJ':(7,4),'NY':(7,4),'CA':(7,4),'MD':(7,4),'IL':(7,-11),'FL':(7,4),
        'NV':(-14,5),'CO':(-15,-11),'CT':(7,4),
    }
    for i, row in sub.iterrows():
        if row['abbr'] in permissive_offsets:
            dx, dy = permissive_offsets[row['abbr']]
            ax_sc.annotate(row['abbr'], (x_j[i], y_j[i]),
                           xytext=(dx, dy), textcoords='offset points',
                           fontsize=8.5, color='#222')
    ax_sc.set_xlabel('% of counties with no abortion clinic (2020)', fontsize=10)
    ax_sc.set_ylabel('Abortions per 1,000 women aged 15-44', fontsize=10)
    ax_sc.set_xlim(-5, 118)
    ax_sc.set_ylim(0, 36)
    ax_sc.grid(alpha=0.25)
    r_p = mpatches.Patch(color=RED,  label='Restrictive-law states')
    o_p = mpatches.Patch(color=GRAY, label='Other states')
    ax_sc.legend(handles=[r_p, o_p], loc='upper right', frameon=False, fontsize=9)

    fig.text(0.5, 0.012, 'Source: Guttmacher Institute, State Facts About Abortion (2020).',
             ha='center', fontsize=8, color='#777')
    fig.savefig(out, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


# ============================================================
# Final AGAINST composite (Side B)
# ============================================================
def build_against_image(out='against.png'):
    fig = plt.figure(figsize=(14, 13.5))
    fig.patch.set_facecolor('white')

    fig.text(0.5, 0.975, 'SIDE B: refuting the proposition',
             ha='center', fontsize=11, color=GRAY, fontweight='bold')

    # Section 1: tile map
    fig.text(0.04, 0.945, 'Women in Restrictive States Just Cross State Lines',
             fontsize=15.5, fontweight='bold')
    fig.text(0.04, 0.928,
             '% of state residents who got an abortion and traveled out of state for it (2020). '
             'In Missouri, 99% left the state. In Wyoming, 88%. In South Dakota, 84%.',
             fontsize=10, color='#555')
    ax_map = fig.add_axes([0.06, 0.62, 0.88, 0.28])
    tile_map(ax_map, df[TRAVEL], vmin=0, vmax=100, value_fmt='{:.0f}%')
    add_dual_legend(fig, [0.37, 0.585, 0.26, 0.018], 0, 100,
                    '% of residents who traveled out of state for care',
                    fmt='{:.0f}%')

    # Section 2: travel-rate bar chart (back from the checkpoint)
    fig.text(0.04, 0.55, 'Top exporters: where residents leave for care',
             fontsize=13, fontweight='bold')
    fig.text(0.04, 0.535,
             'Top 15 states ranked by the share of residents who traveled out of state for an abortion.',
             fontsize=9.5, color='#555')
    ax_bar = fig.add_axes([0.16, 0.32, 0.76, 0.20])
    travel_bar_chart(ax_bar)

    # Section 3: gap chart
    fig.text(0.04, 0.275, 'Restrictive States Lose Residents to Their Neighbors',
             fontsize=13, fontweight='bold')
    fig.text(0.04, 0.260,
             'Difference between rate by residence and rate by occurrence. Positive = state exports patients; negative = state imports them.',
             fontsize=9.5, color='#555')
    ax_g = fig.add_axes([0.18, 0.05, 0.74, 0.20])
    diverging_gap(ax_g)

    fig.text(0.5, 0.010, 'Source: Guttmacher Institute, State Facts About Abortion (2020).',
             ha='center', fontsize=8, color='#777')
    fig.savefig(out, dpi=180, facecolor='white')
    plt.close(fig)
    print(f'wrote {out}')


# ============================================================
# Checkpoint PDF (kept for record; uses the same composites)
# ============================================================
def build_pdf(path='checkpoint.pdf'):
    """Older 3-page PDF: keeps existing checkpoint output."""
    build_for_image('for.png')
    build_against_image('against.png')
    print(f'(skipping PDF rebuild; checkpoint.pdf is the prior artifact)')


if __name__ == '__main__':
    build_for_image('for.png')
    build_against_image('against.png')
