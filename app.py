
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
    page_title="LOURE IA",
    page_icon="💙",
    layout="wide",
)

st.title("💙 LOURE IA")
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
    st.header("💙 LOURE IA")
    st.write("¿En qué podemos trabajar hoy?")

    modulo = st.selectbox(
        "Selecciona un módulo",
        [
            "Chat con LOURE IA",
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

    st.divider()
    st.subheader("Información académica")

    integrante1 = st.text_input(
        "Integrante 1",
        placeholder="Nombre completo",
    )
    integrante2 = st.text_input(
        "Integrante 2",
        placeholder="Nombre completo",
    )
    seccion = st.text_input(
        "Sección",
        placeholder="Ej. A",
    )

    st.divider()
    st.caption("LOURE IA · Aprende a tu ritmo 💙")


# =========================================================
# CHAT INTELIGENTE
# =========================================================

def chat_loure_ia():
    st.header("💬 Conversa con LOURE IA")

    st.write(
        "Pregúntame sobre programación lineal, modelación matemática, "
        "método gráfico, simplex, transporte, asignación o inventarios."
    )

    if "loure_messages" not in st.session_state:
        st.session_state.loure_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Holaaa! Soy LOURE IA. 💙\n\n"
                    "Estoy aquí para ayudarte a aprender Investigación "
                    "de Operaciones. Puedes contarme tu ejercicio, pedirme "
                    "una explicación o decirme en qué paso te atoraste."
                ),
            }
        ]

    if st.button("Limpiar conversación"):
        st.session_state.loure_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Empecemos de nuevo! 💙 ¿Qué necesitas aprender hoy?"
                ),
            }
        ]
        st.rerun()

    for mensaje in st.session_state.loure_messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])

    pregunta = st.chat_input("Escribe tu pregunta aquí...")

    if not pregunta:
        return

    st.session_state.loure_messages.append(
        {"role": "user", "content": pregunta}
    )

    with st.chat_message("user"):
        st.markdown(pregunta)

    
clave = os.getenv("GEMINI_API_KEY", "")

try:
    clave = clave or st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass


    if not clave:
        respuesta = (
            "El chat todavía no está conectado a la API. 💙\n\n"
            "Configura `OPENAI_API_KEY` en los Secrets de Streamlit Cloud. "
            "No publiques tu clave en GitHub ni la escribas directamente "
            "en este archivo."
        )
        with st.chat_message("assistant"):
            st.warning(respuesta)

        st.session_state.loure_messages.append(
            {"role": "assistant", "content": respuesta}
        )
        return

    instrucciones = """
Eres LOURE IA, un asistente educativo en español especializado en
Investigación de Operaciones.

Ayudas con:
- Modelación matemática.
- Programación lineal.
- Método gráfico.
- Método simplex.
- Problemas de transporte y asignación.
- Inventarios y cantidad económica de pedido.
- Ejercicios y práctica académica.

Tu comportamiento:
1. Explica de forma clara, amable y paso a paso.
2. Adapta tus explicaciones a estudiantes principiantes.
3. Identifica variables, función objetivo y restricciones cuando aplique.
4. No inventes datos que falten en el enunciado.
5. Explica las fórmulas y cómo comprobar los resultados.
6. Distingue entre resultados calculados y ejemplos ilustrativos.
7. Responde también preguntas generales, pero prioriza el aprendizaje.
8. No afirmes que ejecutaste cálculos en la aplicación si no lo hiciste.
9. Mantén un tono respetuoso, cercano y educativo.
"""

    mensajes_api = [
        {"role": "system", "content": instrucciones}
    ]

    mensajes_api.extend(
        st.session_state.loure_messages[-20:]
    )

   
try:
    with st.chat_message("assistant"):
        with st.spinner("LOURE IA está pensando... 💭"):
            cliente = genai.Client(api_key=clave)

            historial = []
            for mensaje in st.session_state.loure_messages[-20:]:
                rol = "model" if mensaje["role"] == "assistant" else "user"
                historial.append(
                    types.Content(
                        role=rol,
                        parts=[types.Part.from_text(text=mensaje["content"])]
                    )
                )

            resultado = cliente.models.generate_content(
                model="gemini-2.5-flash",
                contents=historial,
                config=types.GenerateContentConfig(
                    system_instruction=instrucciones,
                    temperature=0.4
                )
            )

            respuesta = resultado.text or "No pude generar una respuesta. Inténtalo de nuevo."

            st.markdown(respuesta)

            st.session_state.loure_messages.append(
                {"role": "assistant", "content": respuesta}
            )

except Exception as e:
    st.error(f"Error al conectar con Gemini: {e}")

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
            value=40.0,
            key="coef_x",
        )

    with col2:
        cy = st.number_input(
            "Coeficiente de y en Z",
            value=30.0,
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
                value=predeterminadas[i][0],
                key=f"ax_{i}",
            )

        with c2:
            b = st.number_input(
                "Coeficiente y",
                value=predeterminadas[i][1],
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
                value=predeterminadas[i][3],
                key=f"limite_{i}",
            )

        restricciones.append((a, b, signo, limite))

    if st.button("Resolver problema", type="primary"):
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

            st.subheader("3. Restricciones")

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
                st.subheader("Información del solucionador")
                st.write(
                    "Se utiliza SciPy con el método HiGHS. "
                    "Esto no muestra las tablas manuales de pivotes."
                )
                st.write(resultado.message)
                st.write(f"Iteraciones reportadas: {resultado.nit}")

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
                    label="Óptimo",
                )

                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title("Restricciones y solución óptima")
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
        "Organiza el enunciado, las variables de decisión, "
        "la función objetivo y las restricciones."
    )

    enunciado = st.text_area(
        "Descripción del problema",
        placeholder="Escribe aquí el enunciado de tu ejercicio...",
    )

    col1, col2 = st.columns(2)

    with col1:
        nombre_x = st.text_input(
            "¿Qué representa x?",
            "Producto A",
        )
        unidad_x = st.text_input(
            "Unidad de x",
            "unidades",
        )

    with col2:
        nombre_y = st.text_input(
            "¿Qué representa y?",
            "Producto B",
        )
        unidad_y = st.text_input(
            "Unidad de y",
            "unidades",
        )

    sentido = st.radio(
        "Objetivo del problema",
        ["Maximizar", "Minimizar"],
        horizontal=True,
    )

    a = st.number_input("Coeficiente de x en Z", value=40.0)
    b = st.number_input("Coeficiente de y en Z", value=30.0)

    st.subheader("Modelo matemático")

    st.write(
        f"x = cantidad de {nombre_x}, medida en {unidad_x}."
    )
    st.write(
        f"y = cantidad de {nombre_y}, medida en {unidad_y}."
    )

    if sentido == "Maximizar":
        st.latex(rf"\text{{Maximizar }} Z = {a:g}x + {b:g}y")
    else:
        st.latex(rf"\text{{Minimizar }} Z = {a:g}x + {b:g}y")

    st.latex(r"x \geq 0,\quad y \geq 0")

    if enunciado:
        st.info(
            "Recuerda identificar los recursos limitados del enunciado "
            "para construir las restricciones."
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
                        value=float(i + j + 1),
                        key=f"asig_{n}_{i}_{j}",
                    )
                    fila.append(valor)

            costos.append(fila)

        if st.button("Resolver asignación"):
            filas, columnas = linear_sum_assignment(
                np.array(costos)
            )

            total = sum(
                costos[i][j]
                for i, j in zip(filas, columnas)
            )

            st.success(f"Costo mínimo total: {total:.2f}")

            for i, j in zip(filas, columnas):
                st.write(
                    f"Persona {i+1} → Tarea {j+1}: "
                    f"costo {costos[i][j]:.2f}"
                )

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
                        value=float(i + j + 1),
                        key=f"trans_c_{origenes}_{destinos}_{i}_{j}",
                    )
                    fila.append(valor)

            matriz.append(fila)

        st.write("Oferta por origen")

        oferta = [
            st.number_input(
                f"Oferta origen {i+1}",
                min_value=0.0,
                value=10.0,
                key=f"oferta_{origenes}_{i}",
            )
            for i in range(origenes)
        ]

        st.write("Demanda por destino")

        demanda = [
            st.number_input(
                f"Demanda destino {j+1}",
                min_value=0.0,
                value=10.0,
                key=f"demanda_{destinos}_{j}",
            )
            for j in range(destinos)
        ]

        if st.button("Resolver transporte"):
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
                st.dataframe(
                    plan,
                    use_container_width=True,
                )
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
            value=1200.0,
        )

        S = st.number_input(
            "Costo por pedido",
            min_value=0.0,
            value=25.0,
        )

        H = st.number_input(
            "Costo de mantenimiento por unidad al año",
            min_value=0.01,
            value=2.0,
        )

        if st.button("Calcular EOQ"):
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

    else:
        demanda = st.number_input(
            "Demanda diaria",
            min_value=0.0,
            value=20.0,
        )

        plazo = st.number_input(
            "Tiempo de entrega en días",
            min_value=0.0,
            value=5.0,
        )

        seguridad = st.number_input(
            "Stock de seguridad",
            min_value=0.0,
            value=0.0,
        )

        if st.button("Calcular punto de reorden"):
            punto = demanda * plazo + seguridad

            st.latex(r"PR=dL+SS")

            st.metric(
                "Punto de reorden",
                f"{punto:.2f} unidades",
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

    if "practica_loure" not in st.session_state:
        st.session_state.practica_loure = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    if st.button("Generar otro ejercicio"):
        st.session_state.practica_loure = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    a, b, x, y = st.session_state.practica_loure

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
            st.success("¡Correcto! 💙")
        else:
            st.error(
                f"Revisa la sustitución: Z = {a}({x}) + {b}({y})."
            )
            st.write(f"El resultado correcto es {correcta}.")


# =========================================================
# MOSTRAR EL MÓDULO SELECCIONADO
# =========================================================

if modulo == "Chat con LOURE IA":
    chat_loure_ia()

elif modulo == "Inicio":
    st.header("¡Bienvenido/a a LOURE IA! 💙")

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
        "Completa los nombres de los integrantes y la sección "
        "en el menú lateral si necesitas esa información."
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
    "💙 LOURE IA · Proyecto académico de Investigación de Operaciones. "
    "Comprueba siempre que el modelo represente correctamente "
    "el enunciado del problema."
)
