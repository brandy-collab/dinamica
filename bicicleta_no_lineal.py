"""
Modelo de la bicicleta con neumáticos no lineales (fórmula mágica por eje).

Acompaña a dinamica_lateral_no_lineal_beamer.tex: de aquí salen las curvas
simuladas de las diapositivas (respuesta a escalón y plano de fases).

Convenios (los mismos que en las diapositivas):
  - ejes del vehículo: x hacia delante, y hacia la izquierda;
  - beta: ángulo de deriva del vehículo; omega: velocidad de guiñada;
  - derivas positivas en un giro a izquierdas:
        alpha_f = delta - beta - a*omega/V
        alpha_r = b*omega/V - beta
  - fuerza de cada eje: F_y = mu*F_z*sin(C*atan(B*alpha)), con
    B = C_eje/(C*mu*F_z), de modo que la pendiente en el origen es la
    rigidez a deriva del eje.

Uso:
    python bicicleta_no_lineal.py          dibuja las figuras con matplotlib
    python bicicleta_no_lineal.py --pgf    escribe datos_pgf.tex con los
                                           \\addplot que van en las diapositivas
"""
from dataclasses import dataclass
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

G = 9.81


@dataclass
class Vehiculo:
    m: float = 1500.0      # masa (kg)
    Iz: float = 2500.0     # inercia de guiñada (kg m^2)
    a: float = 1.2         # c.d.g. - eje delantero (m)
    b: float = 1.5         # c.d.g. - eje trasero (m)
    Cf: float = 80e3       # rigidez a deriva del eje delantero (N/rad)
    Cr: float = 90e3       # rigidez a deriva del eje trasero (N/rad)
    muf: float = 0.90      # adherencia efectiva del eje delantero
    mur: float = 1.00      # adherencia efectiva del eje trasero
    C: float = 1.5         # factor de forma de la fórmula mágica

    # --- magnitudes derivadas -------------------------------------------
    @property
    def L(self):
        return self.a + self.b

    @property
    def Fzf(self):
        return self.m * G * self.b / self.L

    @property
    def Fzr(self):
        return self.m * G * self.a / self.L

    @property
    def K(self):
        """Gradiente de subviraje del modelo lineal, en rad/(m/s^2)."""
        return self.m / self.L * (self.b / self.Cf - self.a / self.Cr)

    # --- neumáticos -----------------------------------------------------
    def Fyf(self, alpha):
        B = self.Cf / (self.C * self.muf * self.Fzf)
        return self.muf * self.Fzf * np.sin(self.C * np.arctan(B * alpha))

    def Fyr(self, alpha):
        B = self.Cr / (self.C * self.mur * self.Fzr)
        return self.mur * self.Fzr * np.sin(self.C * np.arctan(B * alpha))

    def derivas(self, beta, omega, delta, V):
        return delta - beta - self.a * omega / V, self.b * omega / V - beta

    # --- dinámica -------------------------------------------------------
    def f(self, x, delta, V, lineal=False):
        """Derivada del estado x = (beta, omega)."""
        beta, omega = x
        af, ar = self.derivas(beta, omega, delta, V)
        if lineal:
            Ff, Fr = self.Cf * af, self.Cr * ar
        else:
            Ff, Fr = self.Fyf(af), self.Fyr(ar)
        dbeta = (Ff + Fr) / (self.m * V) - omega
        domega = (self.a * Ff - self.b * Fr) / self.Iz
        return np.array([dbeta, domega])

    def simula(self, delta_t, V, T=4.0, x0=(0.0, 0.0), lineal=False, n=401):
        """Integra el modelo. delta_t es una función del tiempo (rad).

        Devuelve t, beta, omega y la aceleración lateral a_y."""
        t = np.linspace(0.0, T, n)
        sol = solve_ivp(lambda tt, x: self.f(x, delta_t(tt), V, lineal),
                        (0.0, T), x0, t_eval=t, rtol=1e-8, atol=1e-10, max_step=0.01)
        beta, omega = sol.y
        ay = np.array([V * (self.f(x, delta_t(tt), V, lineal)[0] + x[1])
                       for tt, x in zip(sol.t, sol.y.T)])
        return sol.t, beta, omega, ay

    # --- giro estacionario ----------------------------------------------
    def derivas_estacionarias(self, ay):
        """Derivas de cada eje para una aceleración lateral ay (hasta el pico)."""
        y = np.asarray(ay) / G
        Bf = self.Cf / (self.C * self.muf * self.Fzf)
        Br = self.Cr / (self.C * self.mur * self.Fzr)
        af = np.tan(np.arcsin(y / self.muf) / self.C) / Bf
        ar = np.tan(np.arcsin(y / self.mur) / self.C) / Br
        return af, ar

    def rigideces_locales(self, ay):
        """Pendiente de cada característica de eje en el punto de trabajo."""
        y = np.asarray(ay) / G
        uf, ur = np.arcsin(y / self.muf), np.arcsin(y / self.mur)
        Cfl = self.Cf * np.cos(uf) * np.cos(uf / self.C) ** 2
        Crl = self.Cr * np.cos(ur) * np.cos(ur / self.C) ** 2
        return Cfl, Crl

    def K_local(self, ay):
        """Gradiente de subviraje local, en rad/(m/s^2)."""
        Cfl, Crl = self.rigideces_locales(ay)
        return self.m / self.L * (self.b / Cfl - self.a / Crl)

    def delta_estacionario(self, ay, R):
        af, ar = self.derivas_estacionarias(ay)
        return self.L / R + af - ar

    # --- estabilidad ----------------------------------------------------
    def equilibrios(self, V, delta=0.0):
        """Puntos de equilibrio (beta, omega) y sus autovalores."""
        sols = []
        for b0 in np.radians(np.linspace(-40, 40, 41)):
            for o0 in np.radians(np.linspace(-60, 60, 25)):
                x, _, ier, _ = fsolve(lambda x: self.f(x, delta, V), [b0, o0], full_output=True)
                if ier == 1 and abs(x[0]) < 1 and not any(np.allclose(x, s, atol=1e-5) for s in sols):
                    sols.append(x)
        res = []
        for x in sols:
            eps = 1e-6
            J = np.array([(self.f(x + np.eye(2)[i] * eps, delta, V)
                           - self.f(x - np.eye(2)[i] * eps, delta, V)) / (2 * eps)
                          for i in range(2)]).T
            w, vec = np.linalg.eig(J)
            res.append((x, w, vec))
        return res


def escalon(delta_grados, t0=0.0):
    """Escalón de volante de delta_grados (en la rueda) en el instante t0."""
    d = np.radians(delta_grados)
    return lambda t: d if t >= t0 else 0.0


VEH_A = Vehiculo()                 # el eje delantero satura antes
VEH_B = Vehiculo(mur=0.80)         # el eje trasero satura antes

# Ventana del plano de fases (grados y grados/s)
BETA_MAX, OMEGA_MAX = 30.0, 60.0


def trayectoria(v, V, x0, T=4.0, sentido=1):
    """Trayectoria en el plano de fases con volante recto, hasta salir de la ventana."""
    def sale(t, x):
        return 1.0 - max(abs(x[0]) / np.radians(BETA_MAX * 1.02),
                         abs(x[1]) / np.radians(OMEGA_MAX * 1.02))
    sale.terminal = True
    sol = solve_ivp(lambda t, x: sentido * v.f(x, 0.0, V), (0, T), x0, max_step=0.005,
                    events=sale, rtol=1e-9, atol=1e-11)
    return np.degrees(sol.y[0]), np.degrees(sol.y[1])


def plano_de_fases(v, V):
    """Trayectorias desde el borde de la ventana, separatrices y puntos de silla."""
    inicios = [(b0, s * OMEGA_MAX) for b0 in np.linspace(-BETA_MAX, BETA_MAX, 9) for s in (-1, 1)]
    inicios += [(s * BETA_MAX, o0) for o0 in np.linspace(-45, 45, 7) for s in (-1, 1)]
    lineas = [trayectoria(v, V, np.radians([b0, o0])) for b0, o0 in inicios]
    separatrices, sillas = [], []
    for x, w, vec in v.equilibrios(V):
        if w.real.max() > 1e-6:                      # punto de silla
            sillas.append(np.degrees(x))
            vs = vec[:, np.argmin(w.real)].real      # dirección de la variedad estable
            for s in (-1, 1):
                separatrices.append(trayectoria(v, V, x + s * 1e-4 * vs, T=6.0, sentido=-1))
    return lineas, separatrices, sillas


# ======================================================================
#  Exportación a pgfplots
# ======================================================================
def _coords(x, y, fx="%.3f", fy="%.3f"):
    return " ".join(f"({fx % a},{fy % b})" for a, b in zip(x, y))


def _remuestrea(x, y, sx, sy, paso=0.045, nmin=6, nmax=60):
    """Remuestreo uniforme en longitud de arco (plano normalizado por sx, sy)."""
    ds = np.hypot(np.diff(x) / sx, np.diff(y) / sy)
    s = np.concatenate([[0], np.cumsum(ds)])
    if s[-1] < 1e-9:
        return x[:1], y[:1]
    n = int(np.clip(s[-1] / paso, nmin, nmax))
    si = np.linspace(0, s[-1], n)
    return np.interp(si, s, x), np.interp(si, s, y)


def datos_pgf(V=100 / 3.6):
    """Devuelve, por figura, las líneas \\addplot que se insertan en el .tex."""
    sang = " " * 10
    out = {}

    # --- escalón: omega/delta, vehículo A ---
    s = ""
    for color, dd in zip(("usnavy", "usnavy!50", "usred", "usgold"), (1, 2, 4, 6)):
        t, be, om, ay = VEH_A.simula(escalon(dd), V, T=3.0, n=601)
        tt = np.concatenate([np.linspace(0, 1.0, 41), np.linspace(1.05, 3.0, 40)])
        gan = np.interp(tt, t, om / np.radians(dd))
        s += f"{sang}\\addplot[{color}] coordinates {{{_coords(tt, gan)}}};\n"
        s += f"{sang}\\addlegendentry{{$\\delta = {dd}^\\circ$}}\n"
    out["escalon"] = s

    # --- mismo escalón en A y B: beta(t) ---
    s = ""
    for nombre, v, color in (("A", VEH_A, "usred"), ("B", VEH_B, "usgold")):
        for dd, trazo in ((2, ""), (4, ",dashed")):
            t, be, om, ay = v.simula(escalon(dd), V, T=3.0, n=601)
            b = np.degrees(be)
            vis = b > -34
            tt, bb = t[vis], b[vis]
            idx = np.unique(np.concatenate([np.arange(0, len(tt), 10), [len(tt) - 1]]))
            s += f"{sang}\\addplot[{color}{trazo}] coordinates {{{_coords(tt[idx], bb[idx], '%.2f', '%.2f')}}};\n"
            s += f"{sang}\\addlegendentry{{Vehículo {nombre}, $\\delta = {dd}^\\circ$}}\n"
    out["beta"] = s

    # --- planos de fases ---
    flecha = ("postaction={decorate},decoration={markings,"
              "mark=at position 0.4 with {\\arrow{Stealth[length=3.5pt,width=3pt]}}}")
    for nombre, v in (("A", VEH_A), ("B", VEH_B)):
        lineas, separatrices, sillas = plano_de_fases(v, V)
        s = ""
        for x, y in lineas:
            if len(x) < 5:
                continue
            xr, yr = _remuestrea(x, y, BETA_MAX, OMEGA_MAX)
            if len(xr) >= 3:
                s += f"{sang}\\addplot[txtgray!75,{flecha}] coordinates {{{_coords(xr, yr, '%.2f', '%.2f')}}};\n"
        for x, y in separatrices:
            xr, yr = _remuestrea(x, y, BETA_MAX, OMEGA_MAX)
            s += f"{sang}\\addplot[usred,line width=0.9pt] coordinates {{{_coords(xr, yr, '%.2f', '%.2f')}}};\n"
        out["fases" + nombre] = s
    return out


# ======================================================================
#  Figuras con matplotlib
# ======================================================================
def figuras(V=100 / 3.6):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    gr = np.degrees

    # 1. características de eje
    al = np.radians(np.linspace(0, 18, 200))
    ax[0, 0].plot(gr(al), VEH_A.Fyf(al) / VEH_A.Fzf, label="Eje delantero")
    ax[0, 0].plot(gr(al), VEH_A.Fyr(al) / VEH_A.Fzr, label="Eje trasero (A)")
    ax[0, 0].plot(gr(al), VEH_B.Fyr(al) / VEH_B.Fzr, "--", label="Eje trasero (B)")
    ax[0, 0].set(xlabel="deriva (grados)", ylabel="$F_y/F_z$", title="Características de eje")
    ax[0, 0].legend()

    # 2. diagrama de maniobrabilidad
    for v, nombre, lim in ((VEH_A, "A", 0.90), (VEH_B, "B", 0.80)):
        y = np.linspace(0, lim * 0.9999, 400)
        af, ar = v.derivas_estacionarias(y * G)
        ax[0, 1].plot(gr(af - ar), y, label=f"Vehículo {nombre}")
    y = np.linspace(0, 1, 2)
    ax[0, 1].plot(gr(VEH_A.K * G * y), y, "k:", label="Modelo lineal")
    ax[0, 1].set(xlim=(-2, 8), xlabel=r"$\alpha_f-\alpha_r$ (grados)", ylabel="$a_y/g$",
                 title="Diagrama de maniobrabilidad")
    ax[0, 1].legend()

    # 3. escalón: omega/delta
    for dd in (1, 2, 4, 6):
        t, be, om, ay = VEH_A.simula(escalon(dd), V, T=3.0)
        ax[0, 2].plot(t, om / np.radians(dd), label=rf"$\delta={dd}^\circ$")
    t, be, om, ay = VEH_A.simula(escalon(1), V, T=3.0, lineal=True)
    ax[0, 2].plot(t, om / np.radians(1), "k:", label="Lineal")
    ax[0, 2].set(xlabel="t (s)", ylabel=r"$\omega/\delta$ (1/s)", title="Escalón, vehículo A")
    ax[0, 2].legend()

    # 4. mismo escalón en A y B
    for v, nombre, color in ((VEH_A, "A", "C0"), (VEH_B, "B", "C1")):
        for dd, trazo in ((2, "-"), (4, "--")):
            t, be, om, ay = v.simula(escalon(dd), V, T=3.0)
            ax[1, 0].plot(t, gr(be), color + trazo, label=rf"{nombre}, $\delta={dd}^\circ$")
    ax[1, 0].set(ylim=(-30, 2), xlabel="t (s)", ylabel=r"$\beta$ (grados)",
                 title="Mismo escalón en A y B")
    ax[1, 0].legend()

    # 5 y 6. planos de fases
    for v, nombre, eje in ((VEH_A, "A", ax[1, 1]), (VEH_B, "B", ax[1, 2])):
        lineas, separatrices, sillas = plano_de_fases(v, V)
        for x, y in lineas:
            eje.plot(x, y, "0.4", lw=0.6)
        for x, y in separatrices:
            eje.plot(x, y, "r", lw=1.2)
        for x in sillas:
            eje.plot(*x, "ro", mfc="w")
        eje.plot(0, 0, "ko")
        eje.set(xlim=(-BETA_MAX, BETA_MAX), ylim=(-OMEGA_MAX, OMEGA_MAX),
                xlabel=r"$\beta$ (grados)", ylabel=r"$\omega$ (grados/s)",
                title=f"Plano de fases, vehículo {nombre}")
    fig.tight_layout()
    fig.savefig("bicicleta_no_lineal.png", dpi=110)
    plt.show()


if __name__ == "__main__":
    if "--pgf" in sys.argv:
        bloques = datos_pgf()
        with open("datos_pgf.tex", "w", encoding="utf-8") as fh:
            for nombre, texto in bloques.items():
                fh.write(f"% --- {nombre} ---\n{texto}\n")
        print("Escrito datos_pgf.tex")
    else:
        figuras()
