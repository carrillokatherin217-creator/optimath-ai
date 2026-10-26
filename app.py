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
            st.success("¡Correcto! 💙")
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
