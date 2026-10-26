"""
Aplicación Streamlit: modelo de la bicicleta con neumáticos lineales y no lineales.

Ejecutar con:
    streamlit run app_bicicleta.py

Usa los módulos bicicleta_no_lineal.py (modelo) y ensayos.py (ensayos con conductor),
que deben estar en la misma carpeta.

Notación (la misma que en las diapositivas de dinámica lateral):
  V_x            velocidad de avance (constante)
  beta = V_y/V_x ángulo de deriva del vehículo
  omega          velocidad de guiñada;  psi: ángulo de guiñada
  C_af, C_ar     rigidez a deriva de cada eje
  mu_f, mu_r     adherencia efectiva de cada eje
  K_us           gradiente de subviraje: delta = L/R + K_us * V_x^2/(g R)
  V_car, V_cr    velocidades característica y crítica
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bicicleta_no_lineal import G, Vehiculo, plano_de_fases
from ensayos import (carriles_dlc, comprueba_dlc, esquinas, resultado_escalon,
                     resultado_radio, simula_ensayo, y_ref_dlc)

st.set_page_config(page_title="Modelo de la bicicleta", layout="wide")

COL_NL = "#2A2275"     # no lineal (azul US)
COL_LIN = "#7A1712"    # lineal (granate US)
COL_3 = "#D4910F"      # tercer trazo (oro US)

# ======================================================================
#  Vehículos predefinidos
# ======================================================================
PRESETS = {
    "Vehículo A (satura antes el eje delantero)":
        dict(m=1500.0, Iz=2500.0, a=1.2, b=1.5, Cf=80.0, Cr=90.0, muf=0.90, mur=1.00, C=1.5),
    "Vehículo B (satura antes el eje trasero)":
        dict(m=1500.0, Iz=2500.0, a=1.2, b=1.5, Cf=80.0, Cr=90.0, muf=0.90, mur=0.80, C=1.5),
    "Ejemplo 1 de clase (sobrevirador)":
        dict(m=900.0, Iz=1800.0, a=1.3, b=1.2, Cf=140.0, Cr=50.0, muf=1.00, mur=1.00, C=1.5),
}


def aplica_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


if "m" not in st.session_state:
    st.session_state["preset"] = list(PRESETS)[0]
    aplica_preset()

# ======================================================================
#  Barra lateral: parámetros
# ======================================================================
with st.sidebar:
    st.header("Vehículo")
    st.selectbox("Vehículo predefinido", list(PRESETS), key="preset", on_change=aplica_preset)

    st.subheader("Masa y geometría")
    st.number_input("Masa, m (kg)", 300.0, 5000.0, step=50.0, key="m")
    st.number_input("Inercia de guiñada, I_z (kg·m²)", 300.0, 10000.0, step=100.0, key="Iz")
    st.number_input("c.d.g. – eje delantero, a (m)", 0.3, 3.0, step=0.05, key="a")
    st.number_input("c.d.g. – eje trasero, b (m)", 0.3, 3.0, step=0.05, key="b")

    st.subheader("Neumáticos (por eje)")
    st.number_input("Rigidez delantera, C_αf (kN/rad)", 10.0, 400.0, step=5.0, key="Cf")
    st.number_input("Rigidez trasera, C_αr (kN/rad)", 10.0, 400.0, step=5.0, key="Cr")
    st.slider("Adherencia delantera, μ_f", 0.1, 1.5, step=0.05, key="muf")
    st.slider("Adherencia trasera, μ_r", 0.1, 1.5, step=0.05, key="mur")
    st.slider("Factor de forma de la fórmula mágica, C", 1.1, 1.9, step=0.05, key="C")

    st.subheader("Velocidad")
    st.slider("Velocidad de avance, V_x (km/h)", 10, 200, 100, step=5, key="Vkmh")
    st.caption("Se usa en el escalón de volante y en el plano de fases. La prueba de radio "
               "constante y el doble cambio de carril tienen su propia velocidad.")

ss = st.session_state
PAR = (ss.m, ss.Iz, ss.a, ss.b, ss.Cf * 1e3, ss.Cr * 1e3, ss.muf, ss.mur, ss.C)
veh = Vehiculo(*PAR)
Vx = ss.Vkmh / 3.6

# ======================================================================
#  Magnitudes del modelo lineal
# ======================================================================
L = veh.L
Wf, Wr = veh.Fzf, veh.Fzr
Kus = Wf / veh.Cf - Wr / veh.Cr                 # rad (por g)
Kus_g = np.degrees(Kus)                          # grados por g
xN = veh.Cr / (veh.Cf + veh.Cr) * L
ME = (xN - veh.a) / L
ay_max = G * min(veh.muf, veh.mur)

st.title("Dinámica lateral: el modelo de la bicicleta")
st.caption("Neumático lineal ($F_y = C_\\alpha\\,\\alpha$) y no lineal (fórmula mágica por eje, "
           "$F_y = \\mu F_z \\sin[C \\arctan(B\\alpha)]$ con $BC\\mu F_z = C_\\alpha$). "
           "Derivas positivas en un giro a izquierdas.")

c0, c1, c2, c3, c4, c5 = st.columns(6)
c0.metric("Carga eje delantero, W_f", f"{Wf:.0f} N", f"{Wf / (Wf + Wr) * 100:.0f} % del peso",
          delta_color="off")
c1.metric("Carga eje trasero, W_r", f"{Wr:.0f} N", f"{Wr / (Wf + Wr) * 100:.0f} % del peso",
          delta_color="off")
c2.metric("K_us (lineal)", f"{Kus_g:+.2f} °/g")
if Kus > 1e-9:
    caracter = "Subvirador"
    c3.metric("V_car", f"{np.sqrt(G * L / Kus) * 3.6:.0f} km/h")
elif Kus < -1e-9:
    caracter = "Sobrevirador"
    c3.metric("V_cr", f"{np.sqrt(-G * L / Kus) * 3.6:.0f} km/h")
else:
    caracter = "Neutro"
    c3.metric("V_car / V_cr", "—")
c4.metric("Margen estático", f"{ME * 100:+.1f} %", caracter, delta_color="off")
c5.metric("a_y máxima", f"{ay_max / G:.2f} g",
          "satura el eje delantero" if veh.muf < veh.mur else
          ("satura el eje trasero" if veh.mur < veh.muf else "saturan ambos ejes"),
          delta_color="off")


def figura(alto=420):
    fig = go.Figure()
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      template="plotly_white")
    return fig


tabs = st.tabs(["Neumáticos", "Giro estacionario", "Ensayos",
                "Plano de fases", "Ecuaciones"])

# ======================================================================
#  1. Neumáticos
# ======================================================================
with tabs[0]:
    st.subheader("Característica efectiva de cada eje")
    al = np.radians(np.linspace(0, 20, 300))
    fig = figura()
    fig.add_scatter(mode="lines", x=np.degrees(al), y=veh.Fyf(al) / Wf, name="Eje delantero",
                    line=dict(color=COL_NL, width=3))
    fig.add_scatter(mode="lines", x=np.degrees(al), y=veh.Fyr(al) / Wr, name="Eje trasero",
                    line=dict(color=COL_3, width=3))
    al_l = np.radians(np.linspace(0, 6, 2))
    fig.add_scatter(mode="lines", x=np.degrees(al_l), y=veh.Cf * al_l / Wf, name="Lineal delantero",
                    line=dict(color=COL_NL, dash="dash", width=1.5))
    fig.add_scatter(mode="lines", x=np.degrees(al_l), y=veh.Cr * al_l / Wr, name="Lineal trasero",
                    line=dict(color=COL_3, dash="dash", width=1.5))
    fig.update_xaxes(title="α (grados)")
    fig.update_yaxes(title="F_y / F_z", range=[0, max(veh.muf, veh.mur) * 1.25])
    st.plotly_chart(fig, width="stretch")
    Bf = veh.Cf / (veh.C * veh.muf * Wf)
    Br = veh.Cr / (veh.C * veh.mur * Wr)
    st.markdown(
        f"Deriva de pico: **{np.degrees(np.tan(np.pi / (2 * veh.C)) / Bf):.1f}°** delante y "
        f"**{np.degrees(np.tan(np.pi / (2 * veh.C)) / Br):.1f}°** detrás.  "
        f"Rigidez normalizada $C_\\alpha/F_z$: {veh.Cf / Wf:.1f} y {veh.Cr / Wr:.1f} rad⁻¹.")

# ======================================================================
#  2. Giro estacionario
# ======================================================================
with tabs[1]:
    y = np.linspace(0, ay_max / G * 0.9999, 500)
    af, ar = veh.derivas_estacionarias(y * G)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Diagrama de maniobrabilidad")
        fig = figura()
        fig.add_scatter(mode="lines", x=np.degrees(af - ar), y=y, name="No lineal",
                        line=dict(color=COL_NL, width=3))
        yl = np.linspace(0, ay_max / G, 2)
        fig.add_scatter(mode="lines", x=Kus_g * yl, y=yl, name="Lineal",
                        line=dict(color=COL_LIN, dash="dash"))
        fig.update_xaxes(title="α_f − α_r = δ − L/R (grados)")
        fig.update_yaxes(title="a_y / g")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Gradiente de subviraje local")
        ym = y[:-5]
        Kloc = np.degrees(veh.K_local(ym * G) * G)
        fig = figura()
        fig.add_scatter(mode="lines", x=ym, y=Kloc, name="K_us* (no lineal)", line=dict(color=COL_NL, width=3))
        fig.add_scatter(mode="lines", x=[0, ym[-1]], y=[Kus_g, Kus_g], name="K_us (lineal)",
                        line=dict(color=COL_LIN, dash="dash"))
        fig.update_xaxes(title="a_y / g")
        fig.update_yaxes(title="K_us* (°/g)", range=[min(-10, Kus_g - 2), max(15, Kus_g + 2)])
        st.plotly_chart(fig, width="stretch")

    st.subheader("Ángulo de dirección en una curva de radio constante")
    R = st.slider("Radio de la curva, R (m)", 20, 500, 200, step=10)
    Vlim = np.sqrt(ay_max * R)
    V = np.linspace(0.1, Vlim * 0.9999, 400)
    ay = V ** 2 / R
    d_nl = np.degrees(veh.delta_estacionario(ay, R))
    Vlin = np.linspace(0.1, max(Vlim, 60), 400)
    d_lin = np.degrees(L / R + Kus * Vlin ** 2 / (G * R))
    fig = figura(380)
    fig.add_scatter(mode="lines", x=V * 3.6, y=d_nl, name="No lineal", line=dict(color=COL_NL, width=3))
    fig.add_scatter(mode="lines", x=Vlin * 3.6, y=d_lin, name="Lineal", line=dict(color=COL_LIN, dash="dash"))
    fig.add_vline(x=Vlim * 3.6, line_dash="dot", line_color="gray",
                  annotation_text=f"límite {Vlim * 3.6:.0f} km/h")
    fig.update_xaxes(title="V_x (km/h)")
    fig.update_yaxes(title="δ en la rueda (grados)",
                     range=[min(-2, np.nanmin(d_nl) - 0.5), min(np.nanmax(d_nl), 15) + 1])
    st.plotly_chart(fig, width="stretch")

    if ss.Vkmh / 3.6 < Vlim:
        ay0 = Vx ** 2 / R
        a0f, a0r = veh.derivas_estacionarias(ay0)
        beta0 = veh.b / R - a0r
        st.markdown(
            f"A **{ss.Vkmh} km/h** en R = {R} m: $a_y$ = {ay0 / G:.2f} g, "
            f"$\\alpha_f$ = {np.degrees(a0f):.2f}°, $\\alpha_r$ = {np.degrees(a0r):.2f}°, "
            f"$\\delta$ = {np.degrees(L / R + a0f - a0r):.2f}° "
            f"(lineal: {np.degrees(L / R + Kus * ay0 / G):.2f}°), "
            f"$\\beta$ = {np.degrees(beta0):.2f}°.")
    else:
        st.warning(f"A {ss.Vkmh} km/h se supera el límite de adherencia en esta curva.")

# ======================================================================
#  3. Ensayos
# ======================================================================
@st.cache_data(show_spinner="Simulando el ensayo…")
def ensayo(par, tipo, opciones, lineal):
    return simula_ensayo(par, tipo, opciones, lineal)


@st.cache_data(show_spinner="Buscando la velocidad máxima de paso…")
def vel_max_dlc(par, desp):
    """Velocidad más alta (de 5 en 5 km/h) a la que se supera el doble cambio de carril."""
    v = Vehiculo(*par)
    ultima = None
    for Vk in range(40, 125, 5):
        r = simula_ensayo(par, "dlc", (("V", Vk), ("desp", desp)), False)
        m = comprueba_dlc(r, v.a, v.b, desp=desp)
        if r["trompo"] or np.nanmin(m) < 0:
            return ultima, Vk
        ultima = Vk
    return ultima, None


def series(res, lin, extra=None):
    """Panel 2x2: omega, beta, a_y y delta/derivas."""
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Velocidad de guiñada ω (°/s)", "Deriva del vehículo β (°)",
                                        "Aceleración lateral a_y / g",
                                        "Ángulo de dirección δ y derivas α_f, α_r (°)"))
    gr = np.degrees
    for r, nombre, color, dash in ((res, "No lineal", COL_NL, "solid"),
                                   (lin, "Lineal", COL_LIN, "dash")):
        if r is None:
            continue
        kw = dict(mode="lines", line=dict(color=color, dash=dash, width=2.5), legendgroup=nombre)
        fig.add_scatter(x=r["t"], y=gr(r["omega"]), row=1, col=1, name=nombre, **kw)
        fig.add_scatter(x=r["t"], y=gr(r["beta"]), row=1, col=2, name=nombre, showlegend=False, **kw)
        fig.add_scatter(x=r["t"], y=r["ay"] / G, row=2, col=1, name=nombre, showlegend=False, **kw)
    fig.add_scatter(mode="lines", x=res["t"], y=gr(res["delta"]), row=2, col=2,
                    name="δ (no lineal)", line=dict(color="gray", width=2))
    fig.add_scatter(mode="lines", x=res["t"], y=gr(res["af"]), row=2, col=2,
                    name="α_f (no lineal)", line=dict(color=COL_NL, width=1.5, dash="dot"))
    fig.add_scatter(mode="lines", x=res["t"], y=gr(res["ar"]), row=2, col=2,
                    name="α_r (no lineal)", line=dict(color=COL_3, width=1.5, dash="dot"))
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=2)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10), template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0))
    return fig


def carrocerias(fig, r, a, b, n=10, color=COL_NL):
    """Dibuja la planta del vehículo en n instantes."""
    idx = np.linspace(0, len(r["t"]) - 1, n).astype(int)
    ex, ey = esquinas(r["X"][idx], r["Y"][idx], r["psi"][idx], a, b)
    orden = [0, 1, 3, 2, 0]                        # delante izq., delante dcha., detrás dcha., detrás izq.
    for k in range(len(idx)):
        fig.add_scatter(mode="lines", x=ex[orden, k], y=ey[orden, k], fill="toself",
                        fillcolor="rgba(42,34,117,0.10)", line=dict(color=color, width=1),
                        hoverinfo="skip", showlegend=False)


with tabs[2]:
    ensayo_tipo = st.radio("Ensayo", ["Prueba de radio constante", "Doble cambio de carril",
                                      "Escalón de volante"], horizontal=True)
    comparar = st.checkbox("Comparar con el modelo lineal", value=True)

    # ------------------------------------------------------------------
    if ensayo_tipo == "Prueba de radio constante":
        st.caption("Tipo ISO 4138: el conductor sigue un círculo de radio R mientras la velocidad "
                   "aumenta lentamente. Se registra el ángulo de dirección necesario frente a la "
                   "aceleración lateral.")
        cc = st.columns(4)
        R = cc[0].slider("Radio, R (m)", 20, 300, 100, step=10)
        V0 = cc[1].slider("Velocidad inicial (km/h)", 10, 80, 30, step=5)
        V1 = cc[2].slider("Velocidad final (km/h)", 40, 200, 120, step=5)
        Tr = cc[3].slider("Duración de la rampa (s)", 20, 120, 60, step=10)
        op = (("R", float(R)), ("V0", float(V0)), ("V1", float(max(V1, V0 + 5))), ("Tr", float(Tr)))
        res = ensayo(PAR, "radio", op, False)
        lin = ensayo(PAR, "radio", op, True) if comparar else None

        e, ay_lim, V_lim, sale = resultado_radio(res, R)
        m1, m2, m3 = st.columns(3)
        m1.metric("a_y máxima sobre el círculo", f"{ay_lim:.2f} g")
        m2.metric("Velocidad al salir del círculo", f"{V_lim:.0f} km/h" if sale else "no sale")
        if res["trompo"]:
            m3.metric("Comportamiento en el límite", "Trompo", "sobreviraje límite",
                      delta_color="inverse")
        elif sale and e[-1] > 0:
            m3.metric("Comportamiento en el límite", "Abre la trayectoria", "subviraje límite",
                      delta_color="off")
        elif sale:
            m3.metric("Comportamiento en el límite", "Cierra la trayectoria", "sobreviraje",
                      delta_color="inverse")
        else:
            m3.metric("Comportamiento en el límite", "Sigue en el círculo")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ángulo de dirección frente a aceleración lateral**")
            fig = figura(430)
            dentro = (np.abs(e) < 0.5) & (res["t"] > 3.0)      # sin el transitorio inicial
            fig.add_scatter(mode="lines", x=res["ay"][dentro] / G, y=np.degrees(res["delta"][dentro]),
                            name="Ensayo (no lineal)", line=dict(color=COL_NL, width=3))
            if lin is not None:
                e_l = np.hypot(lin["X"], lin["Y"] - R) - R
                d_l = (np.abs(e_l) < 0.5) & (lin["t"] > 3.0)
                fig.add_scatter(mode="lines", x=lin["ay"][d_l] / G, y=np.degrees(lin["delta"][d_l]),
                                name="Ensayo (lineal)", line=dict(color=COL_LIN, width=2.5, dash="dash"))
            yy = np.linspace(0, ay_max / G * 0.999, 300)
            fig.add_scatter(mode="lines", x=yy, y=np.degrees(veh.delta_estacionario(yy * G, R)),
                            name="Estacionario teórico", line=dict(color="gray", width=1.5, dash="dot"))
            fig.add_hline(y=np.degrees(L / R), line_dash="dot", line_color=COL_3,
                          annotation_text="Ackermann L/R", annotation_position="bottom right")
            fig.update_xaxes(title="a_y / g")
            fig.update_yaxes(title="δ en la rueda (grados)")
            st.plotly_chart(fig, width="stretch")
            st.caption("La pendiente de la curva es el gradiente de subviraje: "
                       "d(δ)/d(a_y/g) = K_us. Sube si el vehículo es subvirador.")
        with col2:
            st.markdown("**Trayectoria**")
            fig = figura(430)
            th = np.linspace(0, 2 * np.pi, 300)
            fig.add_scatter(mode="lines", x=R * np.sin(th), y=R - R * np.cos(th), name="Círculo",
                            line=dict(color=COL_3, width=1.5, dash="dot"))
            fig.add_scatter(mode="lines", x=res["X"], y=res["Y"], name="No lineal",
                            line=dict(color=COL_NL, width=2.5))
            if lin is not None:
                fig.add_scatter(mode="lines", x=lin["X"], y=lin["Y"], name="Lineal",
                                line=dict(color=COL_LIN, width=1.5, dash="dash"))
            fig.update_xaxes(title="X (m)")
            fig.update_yaxes(title="Y (m)", scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, width="stretch")
        with st.expander("Series temporales"):
            st.plotly_chart(series(res, lin), width="stretch")

    # ------------------------------------------------------------------
    elif ensayo_tipo == "Doble cambio de carril":
        st.caption("Geometría inspirada en ISO 3888-1 (desplazamiento lateral 3,5 m, vehículo de "
                   "1,8 m de ancho). El conductor sigue la línea central con anticipación de la "
                   "curvatura y un retraso de 0,1 s en la dirección, a velocidad constante.")
        desp = 3.5
        Vdlc = st.slider("Velocidad de entrada (km/h)", 30, 120, 60, step=5)
        op = (("V", float(Vdlc)), ("desp", desp))
        res = ensayo(PAR, "dlc", op, False)
        lin = ensayo(PAR, "dlc", op, True) if comparar else None
        marg = comprueba_dlc(res, veh.a, veh.b, desp=desp)
        pasa = (not res["trompo"]) and np.nanmin(marg) >= 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resultado a %d km/h" % Vdlc, "Supera" if pasa else
                  ("Trompo" if res["trompo"] else "Derriba conos"))
        nombres = ("Carril de entrada", "Carril desplazado", "Carril de salida")
        for col, nm, mg in zip((m2, m3, m4), nombres, marg):
            col.metric(f"Margen mínimo, {nm.lower()}",
                       "—" if np.isnan(mg) else f"{mg * 100:+.0f} cm")
        if st.checkbox("Buscar la velocidad máxima de paso (de 5 en 5 km/h)"):
            vok, vko = vel_max_dlc(PAR, desp)
            if vok is None:
                st.warning("No supera la prueba ni a 40 km/h.")
            elif vko is None:
                st.success(f"Supera la prueba hasta al menos {vok} km/h.")
            else:
                st.success(f"Velocidad máxima de paso: **{vok} km/h** (falla a {vko} km/h).")

        fig = figura(380)
        for x0, x1, yc, anc in carriles_dlc(desp=desp):
            xs = np.arange(x0, x1 + 0.01, 2.5)
            for lado in (-1, 1):
                fig.add_scatter(mode="markers", x=xs, y=np.full_like(xs, yc + lado * anc / 2), showlegend=False,
                                marker=dict(symbol="triangle-up", size=8, color="#E8740C"),
                                hoverinfo="skip")
        xr = np.linspace(-20, 140, 400)
        fig.add_scatter(mode="lines", x=xr, y=y_ref_dlc(xr, desp), name="Referencia",
                        line=dict(color="gray", width=1, dash="dot"))
        carrocerias(fig, res, veh.a, veh.b, n=14)
        fig.add_scatter(mode="lines", x=res["X"], y=res["Y"], name="c.d.g. (no lineal)",
                        line=dict(color=COL_NL, width=2.5))
        if lin is not None:
            fig.add_scatter(mode="lines", x=lin["X"], y=lin["Y"], name="c.d.g. (lineal)",
                            line=dict(color=COL_LIN, width=1.5, dash="dash"))
        fig.update_xaxes(title="X (m)", range=[-20, 140])
        fig.update_yaxes(title="Y (m)", range=[-4, 8])
        st.plotly_chart(fig, width="stretch")
        st.caption("Triángulos: conos. Ejes deformados para ver mejor el desplazamiento lateral.")
        st.plotly_chart(series(res, lin), width="stretch")

    # ------------------------------------------------------------------
    else:
        st.caption("Tipo ISO 7401: giro rápido del volante a velocidad constante (la de la barra "
                   "lateral) y registro de la respuesta transitoria.")
        cc = st.columns(4)
        amp = cc[0].slider("δ en la rueda (grados)", 0.5, 8.0, 2.0, step=0.5)
        tr = cc[1].slider("Tiempo de subida (s)", 0.05, 0.5, 0.15, step=0.05)
        t0 = 0.5
        T = cc[2].slider("Tiempo simulado (s)", 2.0, 10.0, 5.0, step=0.5)
        cc[3].metric("Volante (relación 16:1)", f"{amp * 16:.0f}°")
        op = (("V", float(ss.Vkmh)), ("delta", float(amp)), ("t0", t0), ("tr", float(tr)), ("T", float(T)))
        res = ensayo(PAR, "escalon", op, False)
        lin = ensayo(PAR, "escalon", op, True) if comparar else None

        if res["trompo"]:
            st.error(f"El vehículo hace un trompo en t = {res['t'][-1]:.2f} s (|β| > 60°).")
        if lin is not None and lin["trompo"]:
            st.warning(f"El modelo lineal diverge: V_x = {ss.Vkmh} km/h supera la velocidad "
                       f"crítica V_cr = {np.sqrt(-G * L / Kus) * 3.6:.0f} km/h.")
        mt = resultado_escalon(res, t0, tr)
        if mt is not None:
            m = st.columns(6)
            m[0].metric("ω estacionaria", f"{np.degrees(mt['w_ss']):.1f} °/s")
            m[1].metric("Ganancia G_δ^ω", f"{mt['w_ss'] / np.radians(amp):.2f} s⁻¹")
            m[2].metric("a_y estacionaria", f"{mt['ay_ss'] / G:.2f} g")
            m[3].metric("t de respuesta (90 %)", f"{mt['t_resp']:.2f} s")
            m[4].metric("Tiempo de pico", f"{mt['t_pico']:.2f} s")
            m[5].metric("Sobreoscilación de ω", f"{mt['sobre']:.0f} %")
            st.caption("Tiempos medidos desde el 50 % del giro de volante. Lineal: "
                       f"G_δ^ω = {Vx / (L + Kus * Vx ** 2 / G):.2f} s⁻¹."
                       if abs(L + Kus * Vx ** 2 / G) > 1e-6 else "")
        st.plotly_chart(series(res, lin), width="stretch")
        fig = figura(360)
        carrocerias(fig, res, veh.a, veh.b, n=12)
        fig.add_scatter(mode="lines", x=res["X"], y=res["Y"], name="No lineal",
                        line=dict(color=COL_NL, width=2.5))
        if lin is not None:
            fig.add_scatter(mode="lines", x=lin["X"], y=lin["Y"], name="Lineal",
                            line=dict(color=COL_LIN, width=1.5, dash="dash"))
        fig.update_xaxes(title="X (m)")
        fig.update_yaxes(title="Y (m)", scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, width="stretch")

# ======================================================================
#  4. Plano de fases
# ======================================================================
@st.cache_data(show_spinner="Calculando el plano de fases…")
def fases(par, Vx):
    return plano_de_fases(Vehiculo(*par), Vx)


with tabs[3]:
    st.subheader(f"Plano de fases β–ω con volante recto, a {ss.Vkmh} km/h")
    lineas, separatrices, sillas = fases(PAR, Vx)
    fig = figura(560)
    for k, (x, yy) in enumerate(lineas):
        fig.add_scatter(x=x, y=yy, mode="lines", line=dict(color="gray", width=1),
                        name="Trayectorias", legendgroup="tray", showlegend=k == 0)
    for k, (x, yy) in enumerate(separatrices):
        fig.add_scatter(x=x, y=yy, mode="lines", line=dict(color=COL_LIN, width=2.5),
                        name="Separatrices", legendgroup="sep", showlegend=k == 0)
    if sillas:
        s = np.array(sillas)
        fig.add_scatter(x=s[:, 0], y=s[:, 1], mode="markers", name="Puntos de silla",
                        marker=dict(color="white", size=10, line=dict(color=COL_LIN, width=2)))
    origen_estable = 1 + Kus * Vx ** 2 / (G * L) > 0
    fig.add_scatter(x=[0], y=[0], mode="markers",
                    name="Marcha recta (estable)" if origen_estable else "Marcha recta (inestable)",
                    marker=dict(color=COL_NL if origen_estable else "white", size=10,
                                line=dict(color=COL_NL, width=2)))
    fig.update_xaxes(title="β (grados)", range=[-30, 30])
    fig.update_yaxes(title="ω (grados/s)", range=[-60, 60])
    st.plotly_chart(fig, width="stretch")
    if not origen_estable:
        st.error(f"La marcha recta es inestable: V_x = {ss.Vkmh} km/h supera la velocidad crítica "
                 f"V_cr = {np.sqrt(-G * L / Kus) * 3.6:.0f} km/h. Cualquier perturbación crece.")
    elif sillas:
        st.info("Hay puntos de silla: solo las perturbaciones dentro de la región limitada por "
                "las separatrices vuelven a la marcha recta.")
    else:
        st.success("Sin puntos de silla en la ventana: cualquier perturbación vuelve a la "
                   "marcha recta.")

# ======================================================================
#  5. Ecuaciones
# ======================================================================
with tabs[4]:
    st.markdown("**Cinemática** (derivas positivas en un giro a izquierdas)")
    st.latex(r"\alpha_f = \delta - \beta - \frac{a\,\omega}{V_x} \qquad "
             r"\alpha_r = \frac{b\,\omega}{V_x} - \beta \qquad \beta = \frac{V_y}{V_x}")
    st.markdown("**Equilibrio** (origen en el c.d.g., $V_x$ constante)")
    st.latex(r"m\,V_x(\dot\beta + \omega) = F_{yf} + F_{yr} \qquad "
             r"I_z\,\dot\omega = a\,F_{yf} - b\,F_{yr}")
    st.markdown("**Neumáticos**")
    st.latex(r"\text{lineal: } F_y = C_\alpha\,\alpha \qquad "
             r"\text{no lineal: } F_y = \mu F_z \sin\!\left[C \arctan(B\alpha)\right],\ "
             r"B = \frac{C_\alpha}{C\,\mu F_z}")
    st.markdown("**Giro estacionario lineal**")
    st.latex(r"\delta = \frac{L}{R} + K_{us}\,\frac{V_x^2}{g\,R} \qquad "
             r"K_{us} = \frac{W_f}{C_{\alpha f}} - \frac{W_r}{C_{\alpha r}} \qquad "
             r"G_\delta^\omega = \frac{V_x}{L + K_{us} V_x^2/g}")
    st.markdown("**Gradiente local** (rigideces locales $C^*$: pendiente de cada eje en el "
                "punto de trabajo)")
    st.latex(r"K_{us}^* = \frac{m g}{L}\left(\frac{b}{C_{\alpha f}^*} - \frac{a}{C_{\alpha r}^*}\right)")
