import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import control

st.set_page_config(page_title="Control System Analyzer", layout="centered")
st.title("Control System Analysis Tool")


def streamlit_transfer_function():
    st.header("Define Your System")
    method = st.radio("Choose Input Method", ["Transfer Function", "State Space"])

    if method == "Transfer Function":
        k = st.text_input("Enter gain k (default is 1)", "1")
        try:
            k = float(k)
        except:
            k = 1.0

        num_str = st.text_input("Enter numerator factors (comma-separated, e.g. s+1,s+2)", "s+1")
        den_str = st.text_input("Enter denominator factors (comma-separated, e.g. s+3,s+4)", "s+3,s+4")

        if st.button("Create Transfer Function"):
            s = sp.symbols('s')
            num_factors = [sp.sympify(f) for f in num_str.split(',')]
            den_factors = [sp.sympify(f) for f in den_str.split(',')]

            num_expr = k * sp.prod(num_factors)
            den_expr = sp.prod(den_factors)

            st.latex(sp.latex(num_expr / den_expr))

            num_coeffs = [float(c) for c in sp.Poly(num_expr, s).all_coeffs()]
            den_coeffs = [float(c) for c in sp.Poly(den_expr, s).all_coeffs()]

            sys = control.TransferFunction(num_coeffs, den_coeffs)
            return sys

    elif method == "State Space":
        A = st.text_area("A matrix (e.g. [[0, 1], [-2, -3]])", "[[0, 1], [-2, -3]]")
        B = st.text_area("B matrix (e.g. [[0], [1]])", "[[0], [1]]")
        C = st.text_area("C matrix (e.g. [[1, 0]])", "[[1, 0]]")
        D = st.text_area("D matrix (e.g. [[0]])", "[[0]]")

        if st.button("Convert to Transfer Function"):
            try:
                A = np.array(eval(A))
                B = np.array(eval(B))
                C = np.array(eval(C))
                D = np.array(eval(D))
                sys = control.ss2tf(A, B, C, D)
                return sys
            except Exception as e:
                st.error(f"Matrix error: {e}")

    return None


def plot_step_response(sys):
    t, y = control.step_response(sys)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title("Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.grid(True)
    st.pyplot(fig)


def plot_impulse_response(sys):
    t, y = control.impulse_response(sys)
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title("Impulse Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    ax.grid(True)
    st.pyplot(fig)


def show_stability(sys):
    poles = control.poles(sys)
    st.write(f"Poles: {poles}")
    if all(np.real(p) < 0 for p in poles):
        st.success("The system is STABLE.")
    else:
        st.error("The system is UNSTABLE.")


def show_time_specs(sys):
    poles = control.poles(sys)
    for p in poles:
        if np.iscomplex(p) and np.real(p) < 0:
            wn = np.abs(p)
            zeta = -np.real(p) / wn
            break
    else:
        wn, zeta = None, None

    if wn and zeta and 0 < zeta < 1:
        Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) * 100
        Tp = np.pi / (wn * np.sqrt(1 - zeta ** 2))
        Ts = 4 / (zeta * wn)
        Tr = (1.8 / wn) if 0.5 < zeta < 0.9 else None

        st.metric("Peak Overshoot (Mp)", f"{Mp:.2f}%")
        st.metric("Peak Time (Tp)", f"{Tp:.4f} s")
        st.metric("Settling Time (Ts)", f"{Ts:.4f} s")
        if Tr:
            st.metric("Rise Time (Tr)", f"{Tr:.4f} s")
    else:
        t, y = control.step_response(sys)
        Mp = (np.max(y) - 1) * 100
        Tp = t[np.argmax(y)]
        within_bounds = np.where(np.abs(y - 1) <= 0.02)[0]
        Ts = t[within_bounds[-1]] if len(within_bounds) > 0 else None
        rise_indices = np.where((y >= 0.1) & (y <= 0.9))[0]
        Tr = t[rise_indices[-1]] - t[rise_indices[0]] if len(rise_indices) > 0 else None

        st.metric("Peak Overshoot (Mp)", f"{Mp:.2f}%")
        st.metric("Peak Time (Tp)", f"{Tp:.4f} s")
        if Ts:
            st.metric("Settling Time (Ts)", f"{Ts:.4f} s")
        if Tr:
            st.metric("Rise Time (Tr)", f"{Tr:.4f} s")


def plot_bode(sys):
    mag, phase, omega = control.bode(sys, Plot=False)
    fig, ax = plt.subplots(2)
    ax[0].semilogx(omega, 20 * np.log10(mag))
    ax[0].set_title("Bode Plot")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[0].grid(True)
    ax[1].semilogx(omega, np.degrees(phase))
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Frequency (rad/s)")
    ax[1].grid(True)
    st.pyplot(fig)


def plot_root_locus(sys):
    fig, ax = plt.subplots()
    control.root_locus(sys, ax=ax, grid=True)
    ax.set_title("Root Locus")
    st.pyplot(fig)


sys = streamlit_transfer_function()
st.header("choose operation")
analysis = st.selectbox("Select an analysis:", [
        "Step Response", "Impulse Response", "Stability", "Time-Domain Specs", "Bode Plot", "Root Locus", "State-Space"])

if sys:
    #analysis = st.selectbox("Select an analysis:", [
        #"Step Response", "Impulse Response", "Stability", "Time-Domain Specs", "Bode Plot", "Root Locus", "State-Space"])

    if analysis == "Step Response":
        plot_step_response(sys)
    elif analysis == "Impulse Response":
        plot_impulse_response(sys)
    elif analysis == "Stability":
        show_stability(sys)
    elif analysis == "Time-Domain Specs":
        show_time_specs(sys)
    elif analysis == "Bode Plot":
        plot_bode(sys)
    elif analysis == "Root Locus":
        plot_root_locus(sys)
    elif analysis == "State-Space":
        ss_sys = control.tf2ss(sys)
        st.write("A matrix:", ss_sys.A)
        st.write("B matrix:", ss_sys.B)
        st.write("C matrix:", ss_sys.C)
        st.write("D matrix:", ss_sys.D)
