
import os
import random

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    st.header("LOUREANY")
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

Reglas de respuesta rápida:
1. Responde en español, de forma clara, directa y concisa.
2. Para preguntas sencillas, usa una respuesta breve.
3. Para ejercicios, explica los pasos esenciales sin repetir información.
4. Identifica variables, función objetivo y restricciones cuando aplique.
5. Si falta un dato, indica exactamente cuál; nunca lo inventes.
6. Distingue resultados calculados de ejemplos ilustrativos.
7. No afirmes haber ejecutado cálculos que no hayas realizado.
"""


# =========================================================
# CHAT INTELIGENTE
# =========================================================

@st.cache_resource
def obtener_cliente_gemini(clave):
    """Reutiliza el cliente de Gemini durante la vida de la aplicación."""
    return genai.Client(api_key=clave)


def chat_loureany():
    st.header("💬 Conversa con LOUREANY")
    st.write(
        "Pregúntame sobre programación lineal, modelación matemática, "
        "método gráfico, simplex, transporte, asignación o inventarios."
    )

    if "loureany_messages" not in st.session_state:
        st.session_state.loureany_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Holaaa! Soy LOUREANY. 💙\\n\\n"
                    "Cuéntame tu ejercicio o qué necesitas aprender."
                ),
            }
        ]

    if st.button("Limpiar conversación"):
        st.session_state.loureany_messages = [
            {
                "role": "assistant",
                "content": "¡Empecemos de nuevo!  ¿Qué necesitas aprender hoy?",
            }
        ]
        st.rerun()

    for mensaje in st.session_state.loureany_messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input("Escribe tu pregunta aquí...")
    if not pregunta or not pregunta.strip():
        return

    pregunta = pregunta.strip()
    st.session_state.loureany_messages.append(
        {"role": "user", "content": pregunta}
    )
    with st.chat_message("user"):
        st.markdown(pregunta)

    clave = os.getenv("GEMINI_API_KEY", "")
    if not clave:
        try:
            clave = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            clave = ""

    if not clave:
        st.error(
            "Falta la clave GEMINI_API_KEY. Configúrala en los Secrets "
            "de Streamlit Cloud."
        )
        return

    try:
        # Solo enviamos los últimos mensajes para reducir el contexto.
        mensajes_recientes = st.session_state.loureany_messages[-8:]
        historial = []
        for mensaje in mensajes_recientes:
            historial.append(
                types.Content(
                    role="model" if mensaje["role"] == "assistant" else "user",
                    parts=[types.Part.from_text(text=mensaje["content"])],
                )
            )

        cliente = obtener_cliente_gemini(clave)

        # Transmite el texto a medida que Gemini lo genera, en vez de
        # esperar a que termine toda la respuesta para mostrarla.
        def fragmentos_de_respuesta():
            flujo = cliente.models.generate_content_stream(
                model="gemini-3.5-flash-lite",
                contents=historial,
                config=types.GenerateContentConfig(
                    system_instruction=instrucciones,
                    max_output_tokens=400,
                ),
            )
            for fragmento in flujo:
                texto = getattr(fragmento, "text", None)
                if texto:
                    yield texto

        with st.chat_message("assistant"):
            with st.spinner("LOUREANY está pensando... "):
                respuesta = st.write_stream(fragmentos_de_respuesta())

        if not isinstance(respuesta, str):
            respuesta = str(respuesta or "")
        if not respuesta.strip():
            respuesta = "No pude generar una respuesta. Inténtalo de nuevo."

        st.session_state.loureany_messages.append(
            {"role": "assistant", "content": respuesta}
        )

    except Exception as e:
        st.error(
            "No se pudo generar la respuesta. Revisa la conexión, "
            "la clave de Gemini y el nombre del modelo."
        )
        print(f"Error de Gemini: {e}")


# =========================================================
# SOLUCIONADOR DE PROGRAMACIÓN LINEAL
# =========================================================

def resolver_modelo(objetivo, restricciones, sentido):
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for a, b, signo, limite in restricciones:
        if signo == "≤":
            A_ub.append([a, b])
            b_ub.append(limite)

        elif signo == "≥":
            A_ub.append([-a, -b])
            b_ub.append(-limite)

        else:
            A_eq.append([a, b])
            b_eq.append(limite)

    coeficientes = np.array(objetivo, dtype=float)

    if sentido == "Maximizar":
        coeficientes = -coeficientes

    resultado = linprog(
        coeficientes,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None), (0, None)],
        method="highs",
    )

    return resultado


def modulo_lineal(grafica=False, simplex=False):
    st.header("📈 Programación lineal")

    st.write(
        "Define la función objetivo y las restricciones de tu modelo."
    )

    sentido = st.radio(
        "Objetivo",
        ["Maximizar", "Minimizar"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        cx = st.number_input(
            "Coeficiente de x en Z",
            value=None,
            key="coef_x",
        )

    with col2:
        cy = st.number_input(
            "Coeficiente de y en Z",
            value=None,
            key="coef_y",
        )

    cantidad = st.selectbox(
        "Número de restricciones",
        [2, 3, 4, 5],
    )

    predeterminadas = [
        (2.0, 1.0, "≤", 100.0),
        (1.0, 2.0, "≤", 80.0),
        (1.0, 0.0, "≤", 60.0),
        (0.0, 1.0, "≤", 50.0),
        (1.0, 1.0, "≤", 70.0),
    ]

    restricciones = []

    for i in range(cantidad):
        st.markdown(f"**Restricción {i + 1}**")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            a = st.number_input(
                "Coeficiente x",
                value=None,
                key=f"ax_{i}",
            )

        with c2:
            b = st.number_input(
                "Coeficiente y",
                value=None,
                key=f"by_{i}",
            )

        with c3:
            signo = st.selectbox(
                "Signo",
                ["≤", "≥", "="],
                key=f"signo_{i}",
            )

        with c4:
            limite = st.number_input(
                "Límite",
                value=None,
                key=f"limite_{i}",
            )

        restricciones.append((a, b, signo, limite))

    if st.button("Resolver problema", type="primary"):
        faltantes = []
        if cx is None:
            faltantes.append("coeficiente de x en Z")
        if cy is None:
            faltantes.append("coeficiente de y en Z")
        for i, (a, b, signo, limite) in enumerate(restricciones, 1):
            if a is None:
                faltantes.append(f"coeficiente x de la restricción {i}")
            if b is None:
                faltantes.append(f"coeficiente y de la restricción {i}")
            if limite is None:
                faltantes.append(f"límite de la restricción {i}")
        if faltantes:
            st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
            return

        if any(a == 0 and b == 0 for a, b, _, _ in restricciones):
            st.error(
                "Cada restricción necesita al menos un coeficiente "
                "distinto de cero."
            )
            return

        try:
            resultado = resolver_modelo(
                [cx, cy],
                restricciones,
                sentido,
            )

            if not resultado.success:
                if resultado.status == 2:
                    st.error("El modelo no tiene solución factible.")
                elif resultado.status == 3:
                    st.error("El problema no tiene un óptimo finito.")
                else:
                    st.error(resultado.message)
                return

            x, y = resultado.x
            z = cx * x + cy * y

            st.success("¡Se encontró una solución óptima!")

            st.subheader("1. Variables de decisión")
            st.write("x: cantidad de la primera variable.")
            st.write("y: cantidad de la segunda variable.")

            st.subheader("2. Función objetivo")
            st.latex(rf"Z = {cx:g}x + {cy:g}y")

            st.subheader("3. Tabla de restricciones")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Restricción": f"R{i}",
                            "Coeficiente de x": a,
                            "Coeficiente de y": b,
                            "Signo": signo,
                            "Límite": limite,
                        }
                        for i, (a, b, signo, limite) in enumerate(restricciones, 1)
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

            for i, (a, b, signo, limite) in enumerate(
                restricciones, 1
            ):
                st.latex(
                    rf"{a:g}x + {b:g}y \ {signo} \ {limite:g}"
                )

            st.latex(r"x \geq 0,\quad y \geq 0")

            st.subheader("4. Comprobación")

            for i, (a, b, signo, limite) in enumerate(
                restricciones, 1
            ):
                valor = a * x + b * y

                if signo == "≤":
                    cumple = valor <= limite + 1e-7
                elif signo == "≥":
                    cumple = valor >= limite - 1e-7
                else:
                    cumple = abs(valor - limite) <= 1e-7

                estado = "Cumple" if cumple else "No cumple"

                st.write(
                    f"R{i}: {a:g}({x:.4f}) + {b:g}({y:.4f}) "
                    f"= {valor:.4f}; límite {signo} {limite:g}. "
                    f"{estado}"
                )

            st.subheader("5. Resultado")

            c1, c2, c3 = st.columns(3)
            c1.metric("Valor de x", f"{x:.4f}")
            c2.metric("Valor de y", f"{y:.4f}")
            c3.metric("Valor de Z", f"{z:.4f}")

            st.info(
                f"Para {sentido.lower()}, la solución es "
                f"x = {x:.4f}, y = {y:.4f}, Z = {z:.4f}."
            )

            if simplex:
                st.subheader("📋 Información del método simplex")
                st.success("¡El solucionador encontró una solución óptima!")
                st.write(
                    "Se utilizó el optimizador HiGHS de SciPy "
                    "para resolver el modelo de programación lineal."
                )
                st.metric(
                    "Iteraciones reportadas por el solucionador",
                    resultado.nit,
                )
                st.info(
                    "Nota académica: esta solución se obtiene mediante "
                    "un optimizador computacional. No se muestran las "
                    "tablas manuales de cada iteración del método simplex."
                )

            if grafica:
                st.subheader("6. Gráfica de restricciones")

                max_x = max(
                    10.0,
                    x * 1.3 + 1,
                    *[
                        limite / a
                        for a, b, signo, limite in restricciones
                        if a > 0 and limite > 0
                    ],
                )

                max_y = max(
                    10.0,
                    y * 1.3 + 1,
                    *[
                        limite / b
                        for a, b, signo, limite in restricciones
                        if b > 0 and limite > 0
                    ],
                )

                max_x = min(max_x * 1.1, 100000)
                max_y = min(max_y * 1.1, 100000)

                xs = np.linspace(0, max_x, 400)
                gx = np.linspace(0, max_x, 200)
                gy = np.linspace(0, max_y, 200)
                X, Y = np.meshgrid(gx, gy)

                factible = np.ones(X.shape, dtype=bool)
                fig, ax = plt.subplots()

                for i, (a, b, signo, limite) in enumerate(
                    restricciones, 1
                ):
                    if b != 0:
                        ys = (limite - a * xs) / b
                        ax.plot(xs, ys, label=f"Restricción {i}")
                    elif a != 0:
                        ax.axvline(
                            limite / a,
                            label=f"Restricción {i}",
                        )

                    lado = a * X + b * Y

                    if signo == "≤":
                        factible &= lado <= limite + 1e-8
                    elif signo == "≥":
                        factible &= lado >= limite - 1e-8
                    else:
                        factible &= np.abs(lado - limite) < 0.05

                ax.contourf(
                    X,
                    Y,
                    factible.astype(int),
                    levels=[0.5, 1.5],
                    alpha=0.2,
                )

                ax.scatter(
                    [x],
                    [y],
                    marker="*",
                    s=180,
                    label="Punto óptimo",
                )

                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)
                ax.set_xlabel("Cantidad de x")
                ax.set_ylabel("Cantidad de y")
                ax.set_title("Gráfica de restricciones y solución óptima")
                ax.grid(True)
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

        except Exception as error:
            st.error(f"No se pudo resolver: {error}")


# =========================================================
# MODELACIÓN MATEMÁTICA
# =========================================================

def modelacion():
    st.header("🧩 Modelación matemática")
    st.write(
        "Completa los datos para organizar el problema, definir las variables "
        "de decisión y construir la función objetivo."
    )

    enunciado = st.text_area(
        "Descripción del problema",
        placeholder="Escribe aquí el enunciado del ejercicio...",
    )

    col1, col2 = st.columns(2)
    with col1:
        nombre_x = st.text_input("¿Qué representa x?", placeholder="Ej. Producto A")
        unidad_x = st.text_input("Unidad de x", placeholder="Ej. unidades")
    with col2:
        nombre_y = st.text_input("¿Qué representa y?", placeholder="Ej. Producto B")
        unidad_y = st.text_input("Unidad de y", placeholder="Ej. unidades")

    sentido = st.radio(
        "Objetivo del problema",
        ["Maximizar", "Minimizar"],
        horizontal=True,
    )
    a = st.number_input("Coeficiente de x en Z", value=None, key="model_a")
    b = st.number_input("Coeficiente de y en Z", value=None, key="model_b")

    if st.button("Construir modelo matemático", type="primary"):
        faltantes = []
        if not enunciado.strip():
            faltantes.append("descripción del problema")
        if not nombre_x.strip():
            faltantes.append("qué representa x")
        if not unidad_x.strip():
            faltantes.append("unidad de x")
        if not nombre_y.strip():
            faltantes.append("qué representa y")
        if not unidad_y.strip():
            faltantes.append("unidad de y")
        if a is None:
            faltantes.append("coeficiente de x en Z")
        if b is None:
            faltantes.append("coeficiente de y en Z")

        if faltantes:
            st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
            return

        st.subheader("Modelo matemático")
        st.write(f"x = cantidad de {nombre_x}, medida en {unidad_x}.")
        st.write(f"y = cantidad de {nombre_y}, medida en {unidad_y}.")
        if sentido == "Maximizar":
            st.latex(rf"\\text{{Maximizar }} Z = {a:g}x + {b:g}y")
        else:
            st.latex(rf"\\text{{Minimizar }} Z = {a:g}x + {b:g}y")
        st.latex(r"x \\geq 0,\\quad y \\geq 0")
        st.info(
            "Agrega las restricciones a partir de los recursos limitados "
            "que aparecen en el enunciado."
        )


# =========================================================
# TRANSPORTE Y ASIGNACIÓN
# =========================================================

def transporte_y_asignacion():
    st.header("🚚 Transporte y asignación")

    tipo = st.radio(
        "Selecciona el problema",
        ["Asignación", "Transporte"],
        horizontal=True,
    )

    if tipo == "Asignación":
        n = st.selectbox(
            "Número de personas y tareas",
            [2, 3, 4, 5],
        )

        costos = []

        st.write("Ingresa el costo de asignar cada persona a cada tarea.")

        for i in range(n):
            cols = st.columns(n)
            fila = []

            for j in range(n):
                with cols[j]:
                    valor = st.number_input(
                        f"P{i+1} - T{j+1}",
                        min_value=0.0,
                        value=None,
                        key=f"asig_{n}_{i}_{j}",
                    )
                    fila.append(valor)

            costos.append(fila)

        if st.button("Resolver asignación"):
            faltantes = [
                f"costo de persona {i+1} y tarea {j+1}"
                for i in range(n)
                for j in range(n)
                if costos[i][j] is None
            ]
            if faltantes:
                st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
                return

            matriz_costos = np.array(costos, dtype=float)
            st.subheader("Tabla de costos")
            st.dataframe(
                pd.DataFrame(
                    matriz_costos,
                    index=[f"Persona {i+1}" for i in range(n)],
                    columns=[f"Tarea {j+1}" for j in range(n)],
                ),
                use_container_width=True,
            )

            filas, columnas = linear_sum_assignment(matriz_costos)
            total = sum(costos[i][j] for i, j in zip(filas, columnas))

            asignaciones = [
                {
                    "Persona": f"Persona {i+1}",
                    "Tarea asignada": f"Tarea {j+1}",
                    "Costo": round(float(costos[i][j]), 2),
                }
                for i, j in zip(filas, columnas)
            ]

            st.subheader("Tabla de asignaciones óptimas")
            st.dataframe(
                pd.DataFrame(asignaciones),
                hide_index=True,
                use_container_width=True,
            )
            st.success(f"Costo mínimo total: {total:.2f}")

    else:
        st.info(
            "Para resolver un problema de transporte se necesitan "
            "los costos unitarios, la oferta de cada origen y la "
            "demanda de cada destino. Comprueba primero que la oferta "
            "total sea igual a la demanda total, o balancea el modelo."
        )

        origenes = st.selectbox("Número de orígenes", [2, 3, 4])
        destinos = st.selectbox("Número de destinos", [2, 3, 4])

        st.write("Matriz de costos unitarios")

        matriz = []

        for i in range(origenes):
            cols = st.columns(destinos)
            fila = []

            for j in range(destinos):
                with cols[j]:
                    valor = st.number_input(
                        f"Costo O{i+1}-D{j+1}",
                        min_value=0.0,
                        value=None,
                        key=f"trans_c_{origenes}_{destinos}_{i}_{j}",
                    )
                    fila.append(valor)

            matriz.append(fila)

        st.subheader("Tabla de costos unitarios")
        st.dataframe(
            pd.DataFrame(
                matriz,
                index=[f"Origen {i+1}" for i in range(origenes)],
                columns=[f"Destino {j+1}" for j in range(destinos)],
            ),
            use_container_width=True,
        )

        st.write("Oferta por origen")

        oferta = [
            st.number_input(
                f"Oferta origen {i+1}",
                min_value=0.0,
                value=None,
                key=f"oferta_{origenes}_{i}",
            )
            for i in range(origenes)
        ]

        st.write("Demanda por destino")

        demanda = [
            st.number_input(
                f"Demanda destino {j+1}",
                min_value=0.0,
                value=None,
                key=f"demanda_{destinos}_{j}",
            )
            for j in range(destinos)
        ]

        if st.button("Resolver transporte"):
            faltantes = []
            for i, fila in enumerate(matriz, 1):
                for j, valor in enumerate(fila, 1):
                    if valor is None:
                        faltantes.append(f"costo de origen {i} a destino {j}")
            for i, valor in enumerate(oferta, 1):
                if valor is None:
                    faltantes.append(f"oferta del origen {i}")
            for j, valor in enumerate(demanda, 1):
                if valor is None:
                    faltantes.append(f"demanda del destino {j}")
            if faltantes:
                st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
                return

            if not np.isclose(sum(oferta), sum(demanda)):
                st.error(
                    "El modelo no está balanceado. La oferta total "
                    "debe ser igual a la demanda total para este "
                    "solucionador."
                )
                return

            c = np.array(matriz, dtype=float).flatten()

            A_eq = []
            b_eq = []

            for i in range(origenes):
                fila = np.zeros((origenes, destinos))
                fila[i, :] = 1
                A_eq.append(fila.flatten())
                b_eq.append(oferta[i])

            for j in range(destinos):
                fila = np.zeros((origenes, destinos))
                fila[:, j] = 1
                A_eq.append(fila.flatten())
                b_eq.append(demanda[j])

            resultado = linprog(
                c,
                A_eq=np.array(A_eq),
                b_eq=np.array(b_eq),
                bounds=(0, None),
                method="highs",
            )

            if resultado.success:
                plan = resultado.x.reshape(origenes, destinos)

                st.success(
                    f"Costo mínimo total: {resultado.fun:.2f}"
                )
                st.subheader("Tabla de cantidades transportadas")
                tabla_plan = pd.DataFrame(
                    np.round(plan, 4),
                    index=[f"Origen {i+1}" for i in range(origenes)],
                    columns=[f"Destino {j+1}" for j in range(destinos)],
                )
                st.dataframe(tabla_plan, use_container_width=True)

                st.subheader("Resumen de envíos")
                envios = []
                for i in range(origenes):
                    for j in range(destinos):
                        envios.append({
                            "Origen": f"Origen {i+1}",
                            "Destino": f"Destino {j+1}",
                            "Costo unitario": float(matriz[i][j]),
                            "Cantidad enviada": round(float(plan[i, j]), 4),
                            "Costo del envío": round(float(plan[i, j] * matriz[i][j]), 4),
                        })
                st.dataframe(pd.DataFrame(envios), hide_index=True, use_container_width=True)
            else:
                st.error(
                    "No se pudo resolver el modelo: "
                    + resultado.message
                )


# =========================================================
# INVENTARIOS
# =========================================================

def inventarios():
    st.header("📦 Inventarios")

    tipo = st.radio(
        "Selecciona un cálculo",
        [
            "Cantidad económica de pedido (EOQ)",
            "Punto de reorden",
        ],
        horizontal=True,
    )

    if tipo == "Cantidad económica de pedido (EOQ)":
        st.write(
            "Modelo básico con demanda constante y reposición instantánea."
        )

        D = st.number_input(
            "Demanda anual (unidades)",
            min_value=0.01,
            value=None,
        )

        S = st.number_input(
            "Costo por pedido",
            min_value=0.0,
            value=None,
        )

        H = st.number_input(
            "Costo de mantenimiento por unidad al año",
            min_value=0.01,
            value=None,
        )

        if st.button("Calcular EOQ"):
            faltantes = []
            if D is None:
                faltantes.append("demanda anual")
            if S is None:
                faltantes.append("costo por pedido")
            if H is None:
                faltantes.append("costo de mantenimiento por unidad al año")
            if faltantes:
                st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
                return
            if D <= 0:
                st.error("La demanda anual debe ser mayor que cero.")
                return
            if H <= 0:
                st.error("El costo de mantenimiento debe ser mayor que cero.")
                return
            q = np.sqrt((2 * D * S) / H)

            st.latex(r"Q^*=\sqrt{\frac{2DS}{H}}")
            st.metric(
                "Cantidad económica de pedido",
                f"{q:.2f} unidades",
            )

            st.write(
                f"Pedidos aproximados al año: {D / q:.2f}"
            )

            costo = (D / q) * S + (q / 2) * H

            st.write(
                f"Costo anual de ordenar y mantener: {costo:.2f}"
            )
            st.subheader("Resumen de datos y resultados")
            st.dataframe(
                pd.DataFrame([
                    {"Concepto": "Demanda anual", "Valor": D, "Unidad": "unidades/año"},
                    {"Concepto": "Costo por pedido", "Valor": S, "Unidad": "moneda/pedido"},
                    {"Concepto": "Costo de mantenimiento", "Valor": H, "Unidad": "moneda/unidad/año"},
                    {"Concepto": "Cantidad económica de pedido", "Valor": round(float(q), 4), "Unidad": "unidades"},
                    {"Concepto": "Costo anual de ordenar y mantener", "Valor": round(float(costo), 4), "Unidad": "moneda/año"},
                ]),
                hide_index=True,
                use_container_width=True,
            )

    else:
        demanda = st.number_input(
            "Demanda diaria",
            min_value=0.0,
            value=None,
        )

        plazo = st.number_input(
            "Tiempo de entrega en días",
            min_value=0.0,
            value=None,
        )

        seguridad = st.number_input(
            "Stock de seguridad",
            min_value=0.0,
            value=None,
        )

        if st.button("Calcular punto de reorden"):
            faltantes = []
            if demanda is None:
                faltantes.append("demanda diaria")
            if plazo is None:
                faltantes.append("tiempo de entrega")
            if seguridad is None:
                faltantes.append("stock de seguridad")
            if faltantes:
                st.error("Faltan estos datos: " + ", ".join(faltantes) + ".")
                return
            punto = demanda * plazo + seguridad

            st.latex(r"PR=dL+SS")

            st.metric(
                "Punto de reorden",
                f"{punto:.2f} unidades",
            )
            st.subheader("Resumen de datos y resultado")
            st.dataframe(
                pd.DataFrame([
                    {"Concepto": "Demanda diaria", "Valor": demanda, "Unidad": "unidades/día"},
                    {"Concepto": "Tiempo de entrega", "Valor": plazo, "Unidad": "días"},
                    {"Concepto": "Stock de seguridad", "Valor": seguridad, "Unidad": "unidades"},
                    {"Concepto": "Punto de reorden", "Valor": round(float(punto), 4), "Unidad": "unidades"},
                ]),
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# EJERCICIOS Y PRÁCTICA
# =========================================================

def ejercicios():
    st.header("📝 Ejercicios y práctica")

    ejemplos = [
        (
            "1. Maximización básica",
            "Maximiza Z = 40x + 30y; "
            "2x + y ≤ 100; x + 2y ≤ 80; x,y ≥ 0.",
        ),
        (
            "2. Maximización de utilidad",
            "Maximiza Z = 5x + 4y; "
            "x + y ≤ 20; 2x + y ≤ 30; x,y ≥ 0.",
        ),
        (
            "3. Minimización",
            "Minimiza Z = 3x + 2y; "
            "x + y ≥ 10; x + 2y ≥ 12; x,y ≥ 0.",
        ),
        (
            "4. Capacidad de producción",
            "Maximiza Z = 8x + 6y; "
            "2x + y ≤ 16; x + 2y ≤ 14; x,y ≥ 0.",
        ),
        (
            "5. Planificación",
            "Maximiza Z = 7x + 9y; "
            "x + y ≤ 12; 2x + y ≤ 16; x,y ≥ 0.",
        ),
        (
            "6. Presupuesto",
            "Maximiza Z = 10x + 5y; "
            "2x + y ≤ 18; x + 3y ≤ 24; x,y ≥ 0.",
        ),
        (
            "7. Mezcla de productos",
            "Maximiza Z = 6x + 8y; "
            "x + 2y ≤ 16; 3x + 2y ≤ 24; x,y ≥ 0.",
        ),
        (
            "8. Dieta simplificada",
            "Minimiza Z = 4x + 5y; "
            "x + y ≥ 8; 2x + y ≥ 10; x,y ≥ 0.",
        ),
        (
            "9. Publicidad",
            "Maximiza Z = 3x + 5y; "
            "x + y ≤ 15; x + 3y ≤ 30; x,y ≥ 0.",
        ),
        (
            "10. Distribución de horas",
            "Maximiza Z = 12x + 10y; "
            "2x + y ≤ 20; x + 2y ≤ 18; x,y ≥ 0.",
        ),
    ]

    seleccionado = st.selectbox(
        "Selecciona un ejercicio",
        [e[0] for e in ejemplos],
    )

    st.info(dict(ejemplos)[seleccionado])

    st.caption(
        "Puedes utilizar estos enunciados para practicar "
        "la formulación de modelos."
    )

    st.divider()
    st.subheader("Práctica rápida")

    if "practica_loureany" not in st.session_state:
        st.session_state.practica_loureany = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    if st.button("Generar otro ejercicio"):
        st.session_state.practica_loureany = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    a, b, x, y = st.session_state.practica_loureany

    st.write(
        f"Si Z = {a}x + {b}y, calcula Z cuando x = {x} e y = {y}."
    )

    respuesta = st.number_input(
        "Tu respuesta",
        value=0.0,
        key="respuesta_loure",
    )

    if st.button("Comprobar respuesta"):
        correcta = a * x + b * y

        if abs(respuesta - correcta) < 1e-7:
            st.success("¡Correcto! ")
        else:
            st.error(
                f"Revisa la sustitución: Z = {a}({x}) + {b}({y})."
            )
            st.write(f"El resultado correcto es {correcta}.")


# =========================================================
# MOSTRAR EL MÓDULO SELECCIONADO
# =========================================================

if modulo == "Chat con LOUREANY":
    chat_loureany()

elif modulo == "Inicio":
    st.header("¡Bienvenido/a a LOUREANY! 💙")

    st.write(
        "Selecciona un módulo del menú lateral para comenzar "
        "a trabajar."
    )

    st.markdown(
        """
        - 💬 Chat inteligente
        - 🧩 Modelación matemática
        - 📈 Programación lineal
        - 📉 Método gráfico
        - 📋 Método simplex
        - 🚚 Transporte y asignación
        - 📦 Inventarios
        - 📝 Ejercicios y práctica
        """
    )

    st.info(
        "Consulta los integrantes y la información académica en el menú lateral."
    )

elif modulo == "Modelación matemática":
    modelacion()

elif modulo == "Programación lineal":
    modulo_lineal()

elif modulo == "Método gráfico":
    st.header("📉 Método gráfico")
    st.write(
        "Resuelve un modelo de dos variables y visualiza "
        "las restricciones y el punto óptimo."
    )
    modulo_lineal(grafica=True)

elif modulo == "Método simplex":
    st.header("📋 Método simplex")
    st.write(
        "La solución se calcula con SciPy/HiGHS. "
        "Este módulo no presenta tablas manuales de cada pivote."
    )
    modulo_lineal(simplex=True)

elif modulo == "Transporte y asignación":
    transporte_y_asignacion()

elif modulo == "Inventarios":
    inventarios()

elif modulo == "Ejercicios y práctica":
    ejercicios()


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.divider()

st.caption(
    "LOUREANY · Proyecto académico de Investigación de Operaciones. "
      
)
