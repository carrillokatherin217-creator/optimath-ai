
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, linear_sum_assignment
import random

st.set_page_config(page_title="OptiMath AI", page_icon="📊", layout="wide")

st.title("📊 OptiMath AI")
st.subheader("Asistente Virtual de Investigación de Operaciones")
st.write("Aprende a modelar y resolver problemas paso a paso.")
st.caption("Universidad Mariano Gálvez de Guatemala · Investigación de Operaciones")

with st.sidebar:
    st.header("Información académica")
    st.text_input("Integrante 1", placeholder="Nombre completo")
    st.text_input("Integrante 2", placeholder="Nombre completo")
    st.text_input("Sección", placeholder="Ej. A")

modulo = st.selectbox(
    "Selecciona un módulo",
    [
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


def resolver_modelo(objetivo, restricciones, sentido):
    """Resuelve un modelo lineal con dos variables."""
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

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

    c = np.array(objetivo, dtype=float)
    if sentido == "Maximizar":
        c = -c

    return linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=[(0, None), (0, None)],
        method="highs",
    )


def modulo_lineal(grafica=False, simplex=False):
    st.header("📈 Programación lineal")

    sentido = st.radio(
        "Objetivo",
        ["Maximizar", "Minimizar"],
        horizontal=True,
        key=f"sentido_{modulo}",
    )

    c1, c2 = st.columns(2)
    with c1:
        cx = st.number_input("Coeficiente de x en Z", value=40.0)
    with c2:
        cy = st.number_input("Coeficiente de y en Z", value=30.0)

    cantidad = st.selectbox(
        "Cantidad de restricciones",
        [2, 3, 4, 5],
        key=f"cantidad_{modulo}",
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
                "Coef. x",
                value=predeterminadas[i][0],
                key=f"a_{modulo}_{i}",
            )
        with c2:
            b = st.number_input(
                "Coef. y",
                value=predeterminadas[i][1],
                key=f"b_{modulo}_{i}",
            )
        with c3:
            signo = st.selectbox(
                "Signo",
                ["≤", "≥", "="],
                key=f"signo_{modulo}_{i}",
            )
        with c4:
            limite = st.number_input(
                "Límite",
                value=predeterminadas[i][3],
                key=f"limite_{modulo}_{i}",
            )

        restricciones.append((a, b, signo, limite))

    if st.button("Resolver problema", type="primary", key=f"resolver_{modulo}"):
        if any(a == 0 and b == 0 for a, b, _, _ in restricciones):
            st.error("Cada restricción necesita al menos un coeficiente distinto de cero.")
            return

        try:
            resultado = resolver_modelo(
                [cx, cy], restricciones, sentido
            )

            if not resultado.success:
                if resultado.status == 2:
                    st.error("El modelo no tiene solución factible.")
                elif resultado.status == 3:
                    st.error("El problema no tiene un óptimo finito: es no acotado.")
                else:
                    st.error(f"No se pudo resolver: {resultado.message}")
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
            for i, (a, b, signo, limite) in enumerate(restricciones, 1):
                st.latex(rf"{a:g}x + {b:g}y \ {signo} \ {limite:g}")

            st.latex(r"x \geq 0,\quad y \geq 0")

            st.subheader("4. Sustitución y comprobación")
            for i, (a, b, signo, limite) in enumerate(restricciones, 1):
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
                    f"= {valor:.4f} {signo} {limite:g}. {estado}"
                )

            st.subheader("5. Evaluación de la función objetivo")
            st.latex(
                rf"Z = {cx:g}({x:.4f}) + {cy:g}({y:.4f}) = {z:.4f}"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Valor óptimo de x", f"{x:.4f}")
            c2.metric("Valor óptimo de y", f"{y:.4f}")
            c3.metric("Valor óptimo de Z", f"{z:.4f}")

            st.info(
                f"Interpretación: para {sentido.lower()}, "
                f"la solución es x = {x:.4f}, y = {y:.4f}, "
                f"con un valor de Z = {z:.4f}."
            )

            if simplex:
                st.subheader("Información del solucionador simplex")
                st.write(
                    "Se utilizó SciPy con el solucionador HiGHS. "
                    "El estado y las iteraciones son los reportados por "
                    "la librería; no son una tabla manual de pivotes."
                )
                st.write(resultado.message)
                st.write(f"Iteraciones reportadas: {resultado.nit}")

            if grafica:
                st.subheader("6. Gráfica de las restricciones")

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
                gx = np.linspace(0, max_x, 250)
                gy = np.linspace(0, max_y, 250)
                X, Y = np.meshgrid(gx, gy)
                factible = np.ones(X.shape, dtype=bool)

                fig, ax = plt.subplots()

                for i, (a, b, signo, limite) in enumerate(restricciones, 1):
                    if b != 0:
                        ys = (limite - a * xs) / b
                        ax.plot(xs, ys, label=f"Restricción {i}")
                    elif a != 0:
                        ax.axvline(limite / a, label=f"Restricción {i}")

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
                ax.scatter([x], [y], marker="*", s=180, label="Óptimo")
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
            st.error(f"Error al resolver el problema: {error}")


def modelacion():
    st.header("🧩 Modelación matemática")
    st.write("Organiza las variables, la función objetivo y las restricciones.")

    st.text_area(
        "Descripción del problema",
        placeholder="Describe la situación del ejercicio...",
    )

    c1, c2 = st.columns(2)
    with c1:
        nombre_x = st.text_input("Variable x representa", "Producto A")
        unidad_x = st.text_input("Unidad de x", "unidades")
    with c2:
        nombre_y = st.text_input("Variable y representa", "Producto B")
        unidad_y = st.text_input("Unidad de y", "unidades")

    sentido = st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True)
    a = st.number_input("Coeficiente de x", value=40.0, key="model_a")
    b = st.number_input("Coeficiente de y", value=30.0, key="model_b")

    st.subheader("Restricciones")
    a1 = st.number_input("R1: coeficiente de x", value=2.0)
    b1 = st.number_input("R1: coeficiente de y", value=1.0)
    s1 = st.selectbox("R1: signo", ["≤", "≥", "="])
    d1 = st.number_input("R1: límite", value=100.0)

    a2 = st.number_input("R2: coeficiente de x", value=1.0)
    b2 = st.number_input("R2: coeficiente de y", value=2.0)
    s2 = st.selectbox("R2: signo", ["≤", "≥", "="])
    d2 = st.number_input("R2: límite", value=80.0)

    if st.button("Construir modelo matemático"):
        st.write(f"x = {nombre_x} ({unidad_x})")
        st.write(f"y = {nombre_y} ({unidad_y})")
        st.latex(
            rf"\{'max' if sentido == 'Maximizar' else 'min'}\ Z = {a:g}x + {b:g}y"
        )
        st.latex(rf"{a1:g}x + {b1:g}y \ {s1} \ {d1:g}")
        st.latex(rf"{a2:g}x + {b2:g}y \ {s2} \ {d2:g}")
        st.latex(r"x \geq 0,\quad y \geq 0")
        st.info("Comprueba que las variables y las restricciones correspondan al enunciado.")


def transporte_y_asignacion():
    st.header("🚚 Transporte y asignación")
    tipo = st.radio("Tipo de problema", ["Transporte", "Asignación"], horizontal=True)

    if tipo == "Transporte":
        n = st.selectbox("Tamaño de la matriz", [2, 3], key="trans_n")
        costos = []

        for i in range(n):
            columnas = st.columns(n)
            fila = []
            for j in range(n):
                with columnas[j]:
                    fila.append(
                        st.number_input(
                            f"Costo O{i+1}-D{j+1}",
                            min_value=0.0,
                            value=float(i + j + 2),
                            key=f"costo_trans_{i}_{j}",
                        )
                    )
            costos.append(fila)

        st.subheader("Oferta y demanda")
        c1, c2 = st.columns(2)
        ofertas, demandas = [], []

        with c1:
            for i in range(n):
                ofertas.append(
                    st.number_input(
                        f"Oferta origen {i+1}",
                        min_value=0.0,
                        value=10.0,
                        key=f"oferta_{i}",
                    )
                )

        with c2:
            for j in range(n):
                demandas.append(
                    st.number_input(
                        f"Demanda destino {j+1}",
                        min_value=0.0,
                        value=10.0,
                        key=f"demanda_{j}",
                    )
                )

        if st.button("Resolver transporte"):
            if abs(sum(ofertas) - sum(demandas)) > 1e-7:
                st.error(
                    "El problema está desbalanceado. "
                    "La oferta total debe igualar la demanda total."
                )
            else:
                matriz = np.array(costos)
                c = matriz.flatten()
                A_eq, b_eq = [], []

                for i in range(n):
                    fila = np.zeros((n, n))
                    fila[i, :] = 1
                    A_eq.append(fila.flatten())
                    b_eq.append(ofertas[i])

                for j in range(n):
                    fila = np.zeros((n, n))
                    fila[:, j] = 1
                    A_eq.append(fila.flatten())
                    b_eq.append(demandas[j])

                res = linprog(
                    c,
                    A_eq=np.array(A_eq),
                    b_eq=np.array(b_eq),
                    bounds=(0, None),
                    method="highs",
                )

                if res.success:
                    asignacion = res.x.reshape(n, n)
                    st.success(f"Costo mínimo total: {res.fun:.2f}")
                    st.write("Plan óptimo de transporte:")
                    st.dataframe(asignacion)
                    st.latex(r"\min C=\sum_i\sum_j c_{ij}x_{ij}")
                    st.write(
                        "Cada celda indica cuántas unidades se envían "
                        "desde un origen hasta un destino."
                    )
                else:
                    st.error(f"No se pudo resolver: {res.message}")

    else:
        n = st.selectbox("Cantidad de personas y tareas", [2, 3], key="asig_n")
        costos = []

        for i in range(n):
            columnas = st.columns(n)
            fila = []
            for j in range(n):
                with columnas[j]:
                    fila.append(
                        st.number_input(
                            f"Costo persona {i+1}, tarea {j+1}",
                            min_value=0.0,
                            value=float(i + j + 1),
                            key=f"costo_asig_{i}_{j}",
                        )
                    )
            costos.append(fila)

        if st.button("Resolver asignación"):
            filas, columnas = linear_sum_assignment(np.array(costos))
            total = sum(costos[i][j] for i, j in zip(filas, columnas))
            st.success(f"Costo mínimo total: {total:.2f}")
            for i, j in zip(filas, columnas):
                st.write(
                    f"Persona {i+1} → Tarea {j+1}: "
                    f"costo = {costos[i][j]:.2f}"
                )


def inventarios():
    st.header("📦 Inventarios")
    tipo = st.radio(
        "Cálculo",
        ["Cantidad económica de pedido (EOQ)", "Punto de reorden"],
        horizontal=True,
    )

    if tipo == "Cantidad económica de pedido (EOQ)":
        st.write("Modelo básico con demanda constante y reposición instantánea.")
        D = st.number_input("Demanda anual (unidades)", min_value=0.01, value=1200.0)
        S = st.number_input("Costo por pedido", min_value=0.0, value=25.0)
        H = st.number_input(
            "Costo de mantenimiento por unidad al año",
            min_value=0.01,
            value=2.0,
        )

        if st.button("Calcular EOQ"):
            q = np.sqrt((2 * D * S) / H)
            st.latex(r"Q^*=\sqrt{\frac{2DS}{H}}")
            st.latex(rf"Q^*=\sqrt{{\frac{{2({D:g})({S:g})}}{{{H:g}}}}}={q:.4f}")
            st.metric("Cantidad económica de pedido", f"{q:.2f} unidades")
            st.write(f"Pedidos aproximados por año: {D/q:.2f}")
            costo = (D/q)*S + (q/2)*H
            st.write(f"Costo anual de ordenar y mantener: {costo:.2f}")

    else:
        demanda = st.number_input("Demanda diaria", min_value=0.0, value=20.0)
        plazo = st.number_input("Tiempo de entrega en días", min_value=0.0, value=5.0)
        seguridad = st.number_input("Stock de seguridad", min_value=0.0, value=0.0)

        if st.button("Calcular punto de reorden"):
            pr = demanda * plazo + seguridad
            st.latex(r"PR=dL+SS")
            st.latex(
                rf"PR=({demanda:g})({plazo:g})+{seguridad:g}={pr:.2f}"
            )
            st.metric("Punto de reorden", f"{pr:.2f} unidades")


def ejercicios():
    st.header("📝 Ejercicios y práctica")

    ejemplos = [
        ("1. Maximización básica", "Maximiza Z=40x+30y; 2x+y≤100; x+2y≤80; x,y≥0."),
        ("2. Maximización de utilidad", "Maximiza Z=5x+4y; x+y≤20; 2x+y≤30; x,y≥0."),
        ("3. Minimización", "Minimiza Z=3x+2y; x+y≥10; x+2y≥12; x,y≥0."),
        ("4. Capacidad de producción", "Maximiza Z=8x+6y; 2x+y≤16; x+2y≤14; x,y≥0."),
        ("5. Planificación", "Maximiza Z=7x+9y; x+y≤12; 2x+y≤16; x,y≥0."),
        ("6. Presupuesto", "Maximiza Z=10x+5y; 2x+y≤18; x+3y≤24; x,y≥0."),
        ("7. Mezcla de productos", "Maximiza Z=6x+8y; x+2y≤16; 3x+2y≤24; x,y≥0."),
        ("8. Dieta simplificada", "Minimiza Z=4x+5y; x+y≥8; 2x+y≥10; x,y≥0."),
        ("9. Publicidad", "Maximiza Z=3x+5y; x+y≤15; x+3y≤30; x,y≥0."),
        ("10. Distribución de horas", "Maximiza Z=12x+10y; 2x+y≤20; x+2y≤18; x,y≥0."),
    ]

    elegido = st.selectbox("Selecciona un ejercicio", [e[0] for e in ejemplos])
    st.info(dict(ejemplos)[elegido])

    st.caption(
        "Estos diez enunciados son ejemplos iniciales. "
        "Ingresa sus coeficientes en Programación lineal o Método gráfico para resolverlos."
    )

    st.divider()
    st.subheader("Práctica con retroalimentación")
    if "practica" not in st.session_state:
        st.session_state.practica = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    if st.button("Generar otra práctica"):
        st.session_state.practica = (
            random.randint(2, 9),
            random.randint(2, 9),
            random.randint(1, 5),
            random.randint(1, 5),
        )

    a, b, x, y = st.session_state.practica
    st.write(f"Si Z = {a}x + {b}y, calcula Z cuando x = {x} e y = {y}.")
    respuesta = st.number_input("Tu respuesta", value=0.0, key="respuesta_practica")

    if st.button("Comprobar respuesta"):
        correcta = a*x + b*y
        if abs(respuesta - correcta) < 1e-7:
            st.success("¡Correcto! Sustituiste los valores correctamente.")
        else:
            st.error(f"Revisa la sustitución: Z = {a}({x}) + {b}({y}).")
            st.write(f"La respuesta correcta es {correcta}.")


if modulo == "Inicio":
    st.header("¡Bienvenido/a!")
    st.write(
        "Selecciona un módulo para modelar, resolver y practicar "
        "problemas de Investigación de Operaciones."
    )
    st.markdown("""
    - 🧩 Modelación matemática
    - 📈 Programación lineal
    - 📉 Método gráfico
    - 📋 Método simplex
    - 🚚 Transporte y asignación
    - 📦 Inventarios
    - 📝 Ejercicios y práctica
    """)
    st.info("Completa los nombres de los integrantes y la sección en el panel lateral.")

elif modulo == "Modelación matemática":
    modelacion()

elif modulo == "Programación lineal":
    modulo_lineal()

elif modulo == "Método gráfico":
    st.header("📉 Método gráfico")
    st.write("Modelo de dos variables con representación de restricciones y óptimo.")
    modulo_lineal(grafica=True)

elif modulo == "Método simplex":
    st.header("📋 Método simplex")
    st.write(
        "El óptimo se calcula mediante SciPy/HiGHS. "
        "Las iteraciones reportadas no equivalen a mostrar cada tabla manual de pivotes."
    )
    modulo_lineal(simplex=True)

elif modulo == "Transporte y asignación":
    transporte_y_asignacion()

elif modulo == "Inventarios":
    inventarios()

elif modulo == "Ejercicios y práctica":
    ejercicios()

st.divider()
st.caption("OptiMath AI · Proyecto académico. Verifica que el modelo represente correctamente el enunciado.")

