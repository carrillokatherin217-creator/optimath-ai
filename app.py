
import os
import random

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, linear_sum_assignment
from google import genai
from google.genai import types


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="LOUREANY",
    page_icon="💙",
    layout="wide",
)

st.title("💙 LOUREANY")
st.subheader("Tu asistente virtual de Investigación de Operaciones")
st.write(
    "Aprende, modela y resuelve problemas paso a paso "
    "con ayuda de la inteligencia artificial."
)
st.caption("Proyecto académico · Investigación de Operaciones")


# =========================================================
# MENÚ LATERAL
# =========================================================

with st.sidebar:
    st.header("💙 LOUREANY")
    st.write("¿En qué podemos trabajar hoy?")

    modulo = st.selectbox(
        "Selecciona un módulo",
        [
            "Chat con LOUREANY",
            "Inicio",
            "Modelación matemática",
            "Programación lineal",
            "Método gráfico",
            "Método simplex",
            "Transporte y asignación",
            "Inventarios",
            "Ejercicios y práctica",
        ],
    )

   

# Información académica fija
st.sidebar.markdown("---")
st.sidebar.subheader("Información académica")

st.sidebar.markdown("**Integrante 1**")
st.sidebar.write("Karla Rebeca Ortiz Pineda")

st.sidebar.markdown("**Integrante 2**")
st.sidebar.write("Katherin Lourdes Carrillo Garcia")

st.sidebar.markdown("**Integrante 3**")
st.sidebar.write("Melany Alejadra Sanchez Díaz")

st.sidebar.markdown("**Carrera**")
st.sidebar.write("Ingeniería en Sistemas")

st.sidebar.markdown("**Grado**")
st.sidebar.write("Sexto ciclo")

st.sidebar.markdown("**Sección**")
st.sidebar.write("A")
st.caption("LOUREANY · Aprende a tu ritmo 💙")


# =========================================================
# INSTRUCCIONES DEL CHAT
# =========================================================

instrucciones = """
Eres LOUREANY, un asistente educativo en español especializado en
Investigación de Operaciones.

Ayudas con:
- Modelación matemática.
- Programación lineal.
- Método gráfico.
- Método simplex.
- Problemas de transporte y asignación.
- Inventarios y cantidad económica de pedido.
- Ejercicios y práctica académica.
