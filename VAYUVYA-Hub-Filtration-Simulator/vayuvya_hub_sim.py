"""
VAYUVYA - hub filtration simulator (Streamlit)

Run:
    pip install streamlit numpy scipy matplotlib
    streamlit run vayuvya_hub_sim.py

What it models (all numbers are design estimates from the abstract, to be
replaced by bench measurements in Round 2):
  1. 2-D porous-media (Darcy) flow through pre-filter -> pleated filter ->
     carbon -> blower throat, including an optional seal-leak gap at the walls
     (unfiltered bypass). Solved with a finite-volume Laplace-type solver.
  2. Blower curve vs. system curve -> operating point (airflow, pressure).
  3. Size-resolved filtration efficiency per stage and for the whole hub.
  4. Clean-air delivery rate (CADR), ACH and room PM decay.
  5. Electrical power of the blower and battery runtime in outage mode.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import brentq

st.set_page_config(page_title="VAYUVYA hub filter simulator", layout="wide")

# ----------------------------------------------------------------------------
# Constants and geometry
# ----------------------------------------------------------------------------
RHO = 1.2          # air density, kg/m3
MU = 1.8e-5        # air viscosity, Pa.s
W = 0.30           # hub cross-section width used in the 2-D slice, m
NX = 120
DX = W / NX        # cell size (square cells), m
Q_NOM = 300 / 3600  # nominal design flow used to linearise media, m3/s

# thickness t (m), face area A (m2), dP = mult * (a*v + b*v^2), v = face velocity m/s
# Coefficients chosen so that total media dP ~ 45 Pa at 300 m3/h (abstract: 30-60 Pa)
LAYERS = [
    dict(name="Washable pre-filter", t=0.010, A=0.12, a=8.0, b=2.0, col="#f2c14e"),
    dict(name="Pleated particle filter", t=0.030, A=0.75, a=230.0, b=300.0, col="#4e9af1"),
    dict(name="Activated carbon", t=0.020, A=0.20, a=15.0, b=15.0, col="#444444"),
]
PLENUM = 0.04      # inlet / outlet plenum height, m
OUT_W = 0.08       # blower throat opening width in the slice, m

N_IN = round(PLENUM / DX)
ROWS = []
r = N_IN
for L in LAYERS:
    n = max(2, round(L["t"] / DX))
    ROWS.append((r, r + n))
    r += n
N_OUT = round(PLENUM / DX)
NY = r + N_OUT
OC0 = (NX - round(OUT_W / DX)) // 2
OC1 = OC0 + round(OUT_W / DX)

GRADES = {
    "MERV 13-like (~85 % @ 0.3 um)": (0.85, 0.8),
    "H11-like (95 %)": (0.95, 1.0),
    "H12-like (99.5 %)": (0.995, 1.3),
    "H13 (99.95 %)": (0.9995, 1.6),
}


def layer_mults(clog, grade_mult):
    """Resistance multipliers: filter loading and chosen pleated grade."""
    return [1 + 2.0 * clog, grade_mult * (1 + 2.5 * clog), 1 + 0.3 * clog]


# ----------------------------------------------------------------------------
# 2-D Darcy flow solver
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def solve_field(gap_mm, clog, grade_mult):
    mults = layer_mults(clog, grade_mult)
    sigma = np.ones((NY, NX))  # open plenum: effectively no resistance
    for L, (r0, r1), m in zip(LAYERS, ROWS, mults):
        R = m * (L["a"] + L["b"] * Q_NOM / L["A"])  # Pa.s/m
        sigma[r0:r1, :] = L["t"] / R
    g = gap_mm / 1000.0
    if g > 0:
        slot = g ** 3 / (12 * MU * DX)  # laminar slot conductance per cell width
        r0, r1 = ROWS[0][0], ROWS[-1][1]
        sigma[r0:r1, 0] = slot
        sigma[r0:r1, -1] = slot

    s = sigma
    cv = 2 * s[:-1] * s[1:] / (s[:-1] + s[1:])
    ch = 2 * s[:, :-1] * s[:, 1:] / (s[:, :-1] + s[:, 1:])
    idx = np.arange(NY * NX).reshape(NY, NX)
    I = np.concatenate([idx[:-1].ravel(), idx[1:].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    J = np.concatenate([idx[1:].ravel(), idx[:-1].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()])
    V = -np.concatenate([cv.ravel(), cv.ravel(), ch.ravel(), ch.ravel()])
    N = NY * NX
    off = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    A = off + sp.diags(-np.asarray(off.sum(axis=1)).ravel())

    fixed = np.zeros((NY, NX), bool)
    pv = np.zeros((NY, NX))
    fixed[0, :] = True
    pv[0, :] = 1.0                 # room-side inlet (normalised pressure 1)
    fixed[-1, OC0:OC1] = True      # blower throat (pressure 0)
    f = fixed.ravel()
    free = ~f
    Af = A.tocsr()
    rhs = -Af[free][:, f] @ pv.ravel()[f]
    pf = spsolve(Af[free][:, free].tocsc(), rhs)
    p = pv.ravel().copy()
    p[free] = pf
    p = p.reshape(NY, NX)

    vyf = cv * (p[:-1] - p[1:]) / DX
    vxf = ch * (p[:, :-1] - p[:, 1:]) / DX
    vy = np.zeros((NY, NX))
    vy[1:-1] = 0.5 * (vyf[:-1] + vyf[1:])
    vy[0], vy[-1] = vyf[0], vyf[-1]
    vx = np.zeros((NY, NX))
    vx[:, 1:-1] = 0.5 * (vxf[:, :-1] + vxf[:, 1:])
    vx[:, 0], vx[:, -1] = vxf[:, 0], vxf[:, -1]

    mid = ROWS[1][0] + (ROWS[1][1] - ROWS[1][0]) // 2
    row = vyf[mid]
    bypass = float((row[0] + row[-1]) / row.sum()) if g > 0 else 0.0
    vmean = float(vy[mid, 6:-6].mean())
    return p, vx, vy, bypass, vmean


# ----------------------------------------------------------------------------
# Filtration efficiency model (simplified single-fibre style)
# ----------------------------------------------------------------------------
def _diff(d, v):
    return d ** (-2 / 3) * (max(v, 0.01) / 0.1) ** (-2 / 3)


def stage_penetrations(d, vs, eta0):
    """Penetration (fraction passing) of pre-filter, pleated, carbon for diameter d (um)."""
    d = np.asarray(d, float)
    A_p = -np.log(1 - eta0) / (_diff(0.3, 0.1) + 8.3 * 0.3 ** 2)
    P_pre = np.exp(-(0.02 * _diff(d, vs[0]) + 0.4 * d ** 2))
    P_ple = np.exp(-A_p * (_diff(d, vs[1]) + 8.3 * d ** 2))
    P_car = np.exp(-(0.05 * _diff(d, vs[2]) + 0.3 * d ** 2))
    return P_pre, P_ple, P_car


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("VAYUVYA hub filtration simulator")
st.caption("Pre-filter -> pleated filter -> activated carbon -> blower, inside the stationary hub. "
           "Design-estimate model for Round 2 planning; replace parameters with bench data.")

with st.sidebar:
    st.header("Operating conditions")
    duty = st.slider("Hub blower speed (%)", 20, 100, 100, 5) / 100
    clog = st.slider("Filter loading (0 = new, 100 = due for service) %", 0, 100, 0, 5) / 100
    gap_mm = st.slider("Seal leak gap at hub wall (mm)", 0.0, 8.0, 1.0, 0.25)
    grade_name = st.selectbox("Pleated filter grade", list(GRADES), index=1)
    eta0, grade_mult = GRADES[grade_name]
    view = st.radio("Flow-field view", ["Air speed", "Pressure"])

    with st.expander("Blower & hub (advanced)"):
        Pmax = st.number_input("Blower shutoff pressure (Pa)", 50.0, 500.0, 150.0, 10.0)
        Qmax_h = st.number_input("Blower free-air flow (m3/h)", 100.0, 1500.0, 500.0, 50.0)
        eta_b = st.slider("Blower overall efficiency", 0.10, 0.60, 0.25, 0.05)
        A_out = st.number_input("Blower throat area (m2)", 0.005, 0.1, 0.02, 0.005, format="%.3f")

    with st.expander("Room & particles"):
        d_part = st.select_slider("Particle diameter (um)", [0.1, 0.3, 0.5, 1.0, 2.5, 5.0], value=0.5)
        room_V = st.number_input("Room volume (m3)", 10.0, 300.0, 67.5, 2.5)
        C0 = st.number_input("Initial indoor PM (ug/m3)", 5.0, 500.0, 150.0, 5.0)
        Cout = st.number_input("Outdoor PM (ug/m3)", 0.0, 500.0, 60.0, 5.0)
        ach_inf = st.number_input("Infiltration / ventilation (ACH)", 0.0, 5.0, 0.5, 0.1)
        k_dep = st.number_input("Natural deposition (1/h)", 0.0, 2.0, 0.1, 0.05)
        mix = st.slider("Room mixing effectiveness", 0.3, 1.0, 0.8, 0.05)

    with st.expander("Power & outage"):
        P_fan_full = st.number_input("Main fan power at full speed (W)", 5.0, 100.0, 28.0, 1.0)
        fan_speed = st.slider("Main-fan speed in outage (%)", 30, 100, 70, 5) / 100
        P_elec = st.number_input("Electronics (W)", 0.0, 20.0, 3.0, 0.5)
        pack_Wh = st.select_slider("Battery pack (Wh)", [71.7, 112.0, 134.4, 179.2], value=112.0)
        duty_out = st.slider("Hub blower in outage (%)", 0, 100, 50, 10) / 100

# ----------------------------------------------------------------------------
# Solve field, then blower/system operating point
# ----------------------------------------------------------------------------
p_n, vx_n, vy_n, bypass, vmean = solve_field(float(gap_mm), float(clog), float(grade_mult))
mults = layer_mults(clog, grade_mult)
Qmax = Qmax_h / 3600


def layer_dps(Q):
    Qf = (1 - bypass) * Q
    return [m * (L["a"] * (Qf / L["A"]) + L["b"] * (Qf / L["A"]) ** 2) for L, m in zip(LAYERS, mults)]


def throat_loss(Q):
    return 1.0 * 0.5 * RHO * (Q / A_out) ** 2


def sys_dp(Q):
    return sum(layer_dps(Q)) + throat_loss(Q)


def blower_dp(Q, dt):
    return Pmax * dt ** 2 * max(0.0, 1 - (Q / (Qmax * dt)) ** 2)


def operating_point(dt):
    if dt <= 0:
        return 0.0, 0.0
    Qs = brentq(lambda q: blower_dp(q, dt) - sys_dp(q), 1e-9, Qmax * dt * 0.9999)
    return Qs, sys_dp(Qs)


Q, dP_tot = operating_point(duty)
Qf = (1 - bypass) * Q
dps = layer_dps(Q)
vs = [Qf / L["A"] for L in LAYERS]
dP_filters = sum(dps)
P_blower = Q * dP_tot / eta_b

P_pre, P_ple, P_car = stage_penetrations(d_part, vs, eta0)
P_filt = float(P_pre * P_ple * P_car)
P_tot = bypass + (1 - bypass) * P_filt
eta_tot = 1 - P_tot
CADR = Q * 3600 * eta_tot
ACH_eff = CADR * mix / room_V

# ----------------------------------------------------------------------------
# Headline metrics
# ----------------------------------------------------------------------------
c = st.columns(6)
c[0].metric("Hub airflow", f"{Q*3600:,.0f} m3/h", f"{Q*3600-300:+.0f} vs 300 target")
c[1].metric("Media dP", f"{dP_filters:,.0f} Pa", "est. 30-60 Pa", delta_color="off")
c[2].metric("Bypass (unfiltered)", f"{bypass*100:.1f} %")
c[3].metric(f"CADR @ {d_part} um", f"{CADR:,.0f} m3/h", f"{CADR-300:+.0f} vs 300 target")
c[4].metric("Effective ACH", f"{ACH_eff:.1f} /h")
c[5].metric("Blower power", f"{P_blower:.1f} W", "target 10-20 W", delta_color="off")

if bypass > 0.15:
    st.warning(f"{bypass*100:.0f}% of the air bypasses the filters through the seal gap. "
               "Fix the sealing before choosing a better filter grade.")
if Q * 3600 < 300:
    st.info("Hub airflow is below the 300 m3/h design target at these settings "
            "(loaded filter, high-grade media or low blower speed).")

tabs = st.tabs(["Airflow through filters", "Fan & system curve", "Efficiency vs size",
                "Room PM decay", "Outage runtime", "Model notes"])

# ----------------------------------------------------------------------------
# Tab 1: flow field
# ----------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2])
    scale = vs[1] / vmean if vmean != 0 else 1.0
    Hm, Wm = NY * DX * 1000, W * 1000
    X = (np.arange(NX) + 0.5) * DX * 1000
    Y = (np.arange(NY) + 0.5) * DX * 1000
    speed = np.hypot(vx_n, vy_n) * scale

    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    if view == "Pressure":
        im = ax.imshow(p_n * dP_tot, extent=[0, Wm, Hm, 0], cmap="magma_r", aspect="equal", alpha=0.9)
        ax.streamplot(X, Y, vx_n, vy_n, color="white", density=1.6, linewidth=0.8, arrowsize=0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Pressure above blower inlet (Pa)")
    else:
        vmax = max(np.percentile(speed, 97), 1e-3)
        strm = ax.streamplot(X, Y, vx_n * scale, vy_n * scale, color=speed, cmap="turbo",
                             norm=Normalize(0, vmax), density=1.8, linewidth=1.1, arrowsize=0.8)
        cb = fig.colorbar(strm.lines, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Air speed (m/s)")
    for L, (r0, r1) in zip(LAYERS, ROWS):
        y0, y1 = r0 * DX * 1000, r1 * DX * 1000
        ax.add_patch(Rectangle((0, y0), Wm, y1 - y0, facecolor=L["col"], alpha=0.30,
                               edgecolor=L["col"], lw=1.5))
        ax.text(Wm / 2, (y0 + y1) / 2, L["name"], ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.add_patch(Rectangle((OC0 * DX * 1000, Hm - 3), (OC1 - OC0) * DX * 1000, 3, color="red"))
    ax.text(Wm / 2, Hm + 8, "blower throat -> filtered air to main fan", ha="center", va="top",
            color="red", fontsize=8)
    ax.text(Wm / 2, -5, "room air in (hub intake)", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, Wm)
    ax.set_ylim(Hm, 0)
    ax.set_xlabel("hub width (mm)")
    ax.set_ylabel("depth along flow (mm)")
    ax.set_title("Cross-section of hub filter stack (2-D slice)")
    left.pyplot(fig, clear_figure=True)
    plt.close(fig)

    names = ["Room air", "After pre-filter", "After pleated", "After carbon", "Hub outlet\n(incl. bypass)"]
    vals = [100, 100 * P_pre, 100 * P_pre * P_ple, 100 * P_filt, 100 * P_tot]
    vals = [float(v) for v in vals]
    fig2, ax2 = plt.subplots(figsize=(5, 3.4))
    bars = ax2.bar(names, vals, color=["#999", "#f2c14e", "#4e9af1", "#444", "#d1495b"])
    ax2.set_ylabel(f"{d_part} um particles remaining (%)")
    ax2.set_yscale("log")
    ax2.set_ylim(max(min(vals) * 0.5, 1e-3), 150)
    for b_, v_ in zip(bars, vals):
        ax2.text(b_.get_x() + b_.get_width() / 2, v_ * 1.1, f"{v_:.2g}%", ha="center", fontsize=8)
    ax2.tick_params(axis="x", labelsize=7)
    right.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    right.markdown("**Pressure drop by layer**")
    right.table({"Layer": [L["name"] for L in LAYERS] + ["Blower throat"],
                 "Face velocity (m/s)": [f"{v:.2f}" for v in vs] + [f"{Q/A_out:.1f}"],
                 "dP (Pa)": [f"{d:.1f}" for d in dps] + [f"{throat_loss(Q):.1f}"]})
    right.caption("Streamlines are the solved Darcy flow in a planar slice. Gaps at the walls "
                  "show how a seal leak diverts air around the media. Speeds are scaled so the "
                  "pleated-media face velocity matches the operating point.")

# ----------------------------------------------------------------------------
# Tab 2: fan and system curve
# ----------------------------------------------------------------------------
with tabs[1]:
    qs = np.linspace(0, Qmax * 1.0, 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dt, ls in [(1.0, "--"), (duty, "-")]:
        ax.plot(qs * 3600, [blower_dp(q, 1.0) if dt == 1.0 else blower_dp(q, dt) for q in qs],
                ls, color="tab:blue", label=f"Blower @ {int(dt*100)}%")
    ax.plot(qs * 3600, [sys_dp(q) for q in qs], color="tab:red", label="Hub system (filters + throat)")
    ax.plot([Q * 3600], [dP_tot], "ko", label=f"Operating point: {Q*3600:.0f} m3/h, {dP_tot:.0f} Pa")
    ax.axvline(300, color="gray", ls=":", label="300 m3/h target")
    ax.set_xlabel("Flow (m3/h)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_ylim(0, Pmax * 1.1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("The free-air rating of a blower does not set clean-air delivery. The intersection "
               "with the real filter + housing curve does, as the abstract states.")

# ----------------------------------------------------------------------------
# Tab 3: efficiency vs size
# ----------------------------------------------------------------------------
with tabs[2]:
    ds = np.logspace(-2, 1, 200)
    a_, b_, c_ = stage_penetrations(ds, vs, eta0)
    eta_f = 1 - a_ * b_ * c_
    eta_all = (1 - bypass) * eta_f
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(ds, 1 - a_, label="Pre-filter", color="#f2c14e")
    ax.semilogx(ds, 1 - b_, label="Pleated", color="#4e9af1")
    ax.semilogx(ds, 1 - c_, label="Carbon (particles only)", color="#444")
    ax.semilogx(ds, eta_f, "k--", label="Stack (no leak)")
    ax.semilogx(ds, eta_all, "r", lw=2, label="Hub incl. bypass")
    ax.axvline(d_part, color="gray", ls=":")
    ax.set_xlabel("Particle diameter (um)")
    ax.set_ylabel("Capture efficiency")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("Efficiency dips near ~0.3 um (most penetrating size). Carbon is shown only for "
               "particle capture; its gas-removal role is not modelled here.")

# ----------------------------------------------------------------------------
# Tab 4: room PM decay
# ----------------------------------------------------------------------------
with tabs[3]:
    t = np.linspace(0, 2, 241)  # hours

    def room(cadr_eff):
        lam = ach_inf + k_dep + cadr_eff / room_V
        Css = ach_inf * Cout / lam
        return Css + (C0 - Css) * np.exp(-lam * t), lam, Css

    C_on, lam_on, Css_on = room(CADR * mix)
    C_off, lam_off, Css_off = room(0.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t * 60, C_off, "gray", label="No hub filtration")
    ax.plot(t * 60, C_on, "tab:green", lw=2, label="With VAYUVYA hub")
    ax.axhline(35, color="orange", ls=":", label="35 ug/m3 reference")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"Indoor PM ({d_part} um) (ug/m3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Steady-state PM (with hub)", f"{Css_on:.1f} ug/m3", f"{Css_on-Css_off:+.1f} vs none")
    c2.metric("Half-life of excess PM", f"{np.log(2)/lam_on*60:.1f} min", f"{np.log(2)/lam_off*60:.0f} min without", delta_color="off")
    c3.metric("Air changes through filter", f"{CADR/room_V:.1f} /h", "target ~4.4", delta_color="off")
    st.caption("Well-mixed single-zone model: dC/dt = ACH_inf (C_out - C) - k C - CADR*mix/V * C. "
               "Real rooms are not perfectly mixed; the mixing slider covers this crudely.")

# ----------------------------------------------------------------------------
# Tab 5: outage runtime
# ----------------------------------------------------------------------------
with tabs[4]:
    Q_o, dP_o = operating_point(duty_out)
    P_bl_o = Q_o * dP_o / eta_b if duty_out > 0 else 0.0
    P_fan_o = P_fan_full * fan_speed ** 3
    P_total = P_fan_o + P_bl_o + P_elec
    usable = pack_Wh * 0.85 * 0.90
    runtime = usable / P_total if P_total > 0 else float("inf")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Main fan", f"{P_fan_o:.1f} W", f"affinity: {P_fan_full:.0f} W x {fan_speed:.2f}^3", delta_color="off")
    c2.metric("Hub blower", f"{P_bl_o:.1f} W", f"{Q_o*3600:.0f} m3/h", delta_color="off")
    c3.metric("Total load", f"{P_total:.1f} W")
    c4.metric("Runtime", f"{runtime:.1f} h", f"{usable:.1f} Wh usable", delta_color="off")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    packs = [71.7, 112.0, 134.4, 179.2]
    ax.bar([f"{x:g} Wh" for x in packs], [x * 0.765 / P_total for x in packs], color="tab:purple")
    ax.set_ylabel("Runtime (h)")
    ax.set_title(f"Runtime at {P_total:.1f} W load (85% usable x 90% path efficiency)")
    ax.grid(alpha=0.3, axis="y")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 6: notes
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
**Governing model**
- *Flow field:* steady Darcy flow, nabla . (sigma nabla p) = 0, with sigma = thickness / resistance for each
  media layer and a very high sigma for open plenums. The wall leak is a laminar slot, sigma = g^3 / (12 mu dx) (a lower bound: real leaks through cracks/pleat edges add turbulent losses).
  Bypass fraction is read from this field and fed to the efficiency and CADR calculation.
- *Media resistance:* dP = m (a v + b v^2) per layer with face velocity v = Q_filtered / A_face.
  Face areas: pre-filter 0.12 m2, pleated 0.75 m2 (from abstract), carbon 0.20 m2.
- *Operating point:* blower dP = Pmax d^2 (1 - (Q / (Qmax d))^2) intersected with the system curve.
- *Efficiency:* single-fibre style diffusion + interception model; pleated grade calibrated to the
  efficiency at 0.3 um and 0.1 m/s. Higher grades are assumed to add resistance (x0.8 / 1.0 / 1.3 / 1.6).
- *CADR* = Q x [1 - (bypass + (1 - bypass) x stack penetration)].

**Assumptions to replace with Round 2 measurements**
- Blower curve (150 Pa / 500 m3/h) and efficiency (25 %) are placeholders; use the real pressure-flow curve.
- Media coefficients a, b, grade multipliers and the loading law are illustrative.
- The 2-D slice is planar, whereas the hub is axisymmetric, so absolute bypass values are indicative only.
  Calibrate the leak gap against a measured smoke or particle-counter leakage test.
- Carbon gas adsorption, humidity effects, hub-outlet mixing with the main fan and acoustics are not modelled.

**Current configuration:** {grade_name}, leak {gap_mm:.2f} mm, loading {clog*100:.0f} %, blower {duty*100:.0f} %.
""")
