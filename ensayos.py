"""
Ensayos normalizados con el modelo de la bicicleta (bicicleta_no_lineal.py):

  - prueba de radio constante (tipo ISO 4138): velocidad creciente en un círculo;
  - doble cambio de carril (geometría inspirada en ISO 3888-1);
  - escalón de volante (tipo ISO 7401).

Los dos primeros necesitan un conductor que siga la trayectoria:
  - radio constante: seguimiento puro (pure pursuit) + anticipación del subviraje
    lineal + acción integral sobre el error radial;
  - doble cambio de carril: anticipación de la curvatura del camino + realimentación
    del error lateral y del error de rumbo.

El ángulo de dirección que ordena el conductor llega a las ruedas con un retraso de
primer orden (TAU_DIR). Estado integrado: beta, omega, psi, X, Y, la integral del error
del conductor y el ángulo de dirección real.
Convenios: X, Y fijos al suelo; giro a izquierdas positivo; derivas positivas en
un giro a izquierdas (los mismos que en las diapositivas).
"""
import numpy as np
from scipy.integrate import solve_ivp

from bicicleta_no_lineal import G, Vehiculo

ANCHO = 1.8                   # anchura del vehículo (m)
VOLADIZO = 0.9                # voladizo delantero y trasero (m)
DELTA_MAX = np.radians(25.0)  # ángulo máximo en la rueda
TAU_DIR = 0.1                 # retraso del conductor y la dirección (s)


# ======================================================================
#  Geometría del doble cambio de carril
# ======================================================================
def carriles_dlc(w=ANCHO, desp=3.5):
    """Tramos con conos: (x inicial, x final, centro del carril, anchura del carril)."""
    return [(0.0, 15.0, 0.0, 1.1 * w + 0.25),
            (45.0, 70.0, desp, 1.2 * w + 0.25),
            (95.0, 125.0, 0.0, 1.3 * w + 0.25)]


def y_ref_dlc(x, desp=3.5):
    """Línea central de referencia: transiciones cosenoidales de 30 m y 25 m."""
    x = np.asarray(x, dtype=float)
    s1 = np.clip((x - 15.0) / 30.0, 0, 1)
    s2 = np.clip((x - 70.0) / 25.0, 0, 1)
    return desp * ((1 - np.cos(np.pi * s1)) - (1 - np.cos(np.pi * s2))) / 2


def curv_ref_dlc(x, desp=3.5):
    """Curvatura de la línea de referencia."""
    x = np.asarray(x, dtype=float)
    d1 = np.zeros_like(x)
    d2 = np.zeros_like(x)
    for x0, l, sg in ((15.0, 30.0, 1), (70.0, 25.0, -1)):
        m = (x > x0) & (x < x0 + l)
        u = np.pi * (x - x0) / l
        d1 += np.where(m, sg * desp / 2 * np.sin(u) * np.pi / l, 0.0)
        d2 += np.where(m, sg * desp / 2 * np.cos(u) * (np.pi / l) ** 2, 0.0)
    return d2 / (1 + d1 ** 2) ** 1.5


def esquinas(X, Y, psi, a, b, w=ANCHO, vol=VOLADIZO):
    """Coordenadas de las cuatro esquinas de la carrocería: arrays (4, n)."""
    lx = np.array([a + vol, a + vol, -(b + vol), -(b + vol)])[:, None]
    ly = np.array([w / 2, -w / 2, w / 2, -w / 2])[:, None]
    c, s = np.cos(psi)[None, :], np.sin(psi)[None, :]
    return X[None, :] + lx * c - ly * s, Y[None, :] + lx * s + ly * c


def comprueba_dlc(res, a, b, w=ANCHO, desp=3.5):
    """Margen mínimo de la carrocería respecto a los conos de cada tramo (m).
    Negativo: la carrocería sale del carril (derriba conos)."""
    ex, ey = esquinas(res["X"], res["Y"], res["psi"], a, b, w)
    margenes = []
    for x0, x1, yc, anc in carriles_dlc(w, desp):
        dentro = (ex >= x0) & (ex <= x1)
        if not dentro.any():
            margenes.append(np.nan)
            continue
        m = anc / 2 - np.abs(ey[dentro] - yc)
        margenes.append(float(m.min()))
    return margenes


# ======================================================================
#  Simulación de un ensayo
# ======================================================================
def simula_ensayo(par, tipo, opciones, lineal=False):
    """
    par: parámetros de Vehiculo (m, Iz, a, b, Cf, Cr, muf, mur, C).
    tipo: 'radio', 'dlc' o 'escalon'.
    opciones (tupla de pares clave-valor, para poder guardarla en caché):
      radio:   R (m), V0 y V1 (km/h), Tr (s, duración de la rampa de velocidad)
      dlc:     V (km/h), desp (m)
      escalon: V (km/h), delta (grados en la rueda), t0 (s), tr (s, tiempo de subida), T (s)
    """
    v = Vehiculo(*par)
    L = v.L
    o = dict(opciones)
    Kus = v.Fzf / v.Cf - v.Fzr / v.Cr              # gradiente lineal (rad por g)

    if tipo == "radio":
        R, V0, V1, Tr = o["R"], o["V0"] / 3.6, o["V1"] / 3.6, o["Tr"]
        T = Tr
        Vfun = lambda t: V0 + (V1 - V0) * min(t / Tr, 1.0)
        x0 = [0.0, V0 / R, 0.0, 0.0, 0.0, 0.0, L / R]
    elif tipo == "dlc":
        Vc = o["V"] / 3.6
        Vfun = lambda t: Vc
        X0 = -20.0
        T = (140.0 - X0) / Vc
        x0 = [0.0, 0.0, 0.0, X0, 0.0, 0.0, 0.0]
    else:
        Vc = o["V"] / 3.6
        Vfun = lambda t: Vc
        T = o["T"]
        x0 = [0.0] * 7

    def delta_de(t, s):
        """Ángulo de dirección ordenado por el conductor (o el escalón)."""
        beta, omega, psi, X, Y, I, _ = s
        V = Vfun(t)
        if tipo == "escalon":
            return np.radians(o["delta"]) * np.clip((t - o["t0"]) / max(o["tr"], 1e-3), 0, 1)
        if tipo == "radio":
            # seguimiento puro hacia un punto del círculo a una distancia d por delante
            d = max(V * 1.0, 4.0)
            Px, Py = X + d * np.cos(psi), Y + d * np.sin(psi)
            e = np.hypot(Px, Py - R) - R           # >0: el círculo queda a la izquierda
            dl = 2 * L * e / d ** 2 + Kus * V ** 2 / (G * R) + 0.01 * I
        else:
            des = o.get("desp", 3.5)
            k = float(curv_ref_dlc(X + V * o.get("Tff", 0.25), des))   # curvatura anticipada
            e_y = Y - float(y_ref_dlc(X, des))                     # error lateral
            pend = float(y_ref_dlc(X + 0.05, des) - y_ref_dlc(X - 0.05, des)) / 0.1
            e_c = psi + beta - np.arctan(pend)                     # error de rumbo
            dl = (L + Kus * V ** 2 / G) * k - o.get("ky", 0.01) * e_y - o.get("kc", 0.5) * e_c
        return float(np.clip(dl, -DELTA_MAX, DELTA_MAX))

    def delta_real(t, s):
        return delta_de(t, s) if tipo == "escalon" else s[6]

    def rhs(t, s):
        beta, omega, psi, X, Y, I, dlt = s
        V = Vfun(t)
        db, dw = v.f((beta, omega), delta_real(t, s), V, lineal)
        Vy = V * beta
        e_rad = np.hypot(X, Y - o["R"]) - o["R"] if tipo == "radio" else 0.0
        return [db, dw, omega, V * np.cos(psi) - Vy * np.sin(psi),
                V * np.sin(psi) + Vy * np.cos(psi), e_rad,
                (delta_de(t, s) - dlt) / TAU_DIR if tipo != "escalon" else 0.0]

    def trompo(t, s):
        return np.radians(60) - abs(s[0])
    trompo.terminal = True

    sol = solve_ivp(rhs, (0, T), x0, max_step=0.01, rtol=1e-7, atol=1e-9, events=trompo)
    t = sol.t
    beta, omega, psi, X, Y, I, _ = sol.y
    Vt = np.array([Vfun(tt) for tt in t])
    dl = np.array([delta_real(tt, s) for tt, s in zip(t, sol.y.T)])
    der = np.array([v.f((b, w), d, V, lineal) for b, w, d, V in zip(beta, omega, dl, Vt)])
    ay = Vt * (der[:, 0] + omega)
    af, ar = v.derivas(beta, omega, dl, Vt)
    return dict(t=t, beta=beta, omega=omega, psi=psi, X=X, Y=Y, delta=dl, ay=ay,
                af=af, ar=ar, V=Vt, trompo=sol.status == 1)


# ======================================================================
#  Resultados de cada ensayo
# ======================================================================
def resultado_radio(res, R, tol=0.5):
    """Error radial y aceleración lateral máxima mientras el vehículo sigue en el círculo."""
    e = np.hypot(res["X"], res["Y"] - R) - R
    fuera = np.abs(e) > tol
    i = int(np.argmax(fuera)) if fuera.any() else len(e)
    ay_lim = float(np.max(res["ay"][:max(i, 1)]) / G)
    V_lim = float(res["V"][max(i - 1, 0)] * 3.6)
    return e, ay_lim, V_lim, fuera.any()


def resultado_escalon(res, t0, tr):
    """Métricas tipo ISO 7401 de la velocidad de guiñada."""
    t, w = res["t"], res["omega"]
    if res["trompo"] or len(t) < 10:
        return None
    w_ss = float(np.mean(w[t >= t[-1] - 0.5]))
    if abs(w_ss) < 1e-6:
        return None
    t50 = t0 + tr / 2                                   # 50 % del giro de volante
    i90 = np.argmax(w >= 0.9 * w_ss) if w_ss > 0 else np.argmax(w <= 0.9 * w_ss)
    ipk = int(np.argmax(w)) if w_ss > 0 else int(np.argmin(w))
    return dict(w_ss=w_ss, t_resp=float(t[i90] - t50), t_pico=float(t[ipk] - t50),
                sobre=float((w[ipk] / w_ss - 1) * 100),
                ay_ss=float(np.mean(res["ay"][t >= t[-1] - 0.5])),
                beta_ss=float(np.mean(res["beta"][t >= t[-1] - 0.5])))
