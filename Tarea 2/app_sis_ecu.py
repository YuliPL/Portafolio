from shiny import App, ui, render, reactive
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Interfaz 
app_ui = ui.page_fluid(
    ui.panel_title("Sistema de Ecuaciones Lineales"),
    ui.input_numeric("n", "Número de incógnitas (2 a 5):", value=2, min=2, max=5),
    ui.output_ui("coef_inputs"),
    ui.output_ui("const_inputs"),
    ui.input_action_button("resolver", "Resolver sistema"),
    ui.output_text_verbatim("resultado", placeholder=True),
    ui.output_image("grafico", width="100%")
)

# Lógica reactiva
def server(input, output, session):

    @output
    @render.ui
    def coef_inputs():
        return ui.TagList(*[
            ui.input_text(f"fila_{i}", f"Coeficientes de la ecuación {i+1} (separados por espacio)", "")
            for i in range(input.n())
        ])

    @output
    @render.ui
    def const_inputs():
        return ui.input_text("terminos", "Términos independientes (separados por espacio)", "")

    @reactive.calc
    @reactive.event(input.resolver)
    def resolver_sistema():
        n = input.n()
        A, b = [], []

        for i in range(n):
            fila_str = input[f"fila_{i}"]().strip()
            fila = list(map(float, fila_str.split()))
            if len(fila) != n:
                raise ValueError(f"La ecuación {i+1} no tiene {n} coeficientes.")
            A.append(fila)

        b_str = input.terminos().strip()
        b = list(map(float, b_str.split()))
        if len(b) != n:
            raise ValueError("El número de términos independientes no coincide con el número de incógnitas.")

        A_np = np.array(A)
        b_np = np.array(b)
        solucion = np.linalg.solve(A_np, b_np)

        return A_np, b_np, solucion

    @output
    @render.text
    def resultado():
        try:
            A_np, b_np, solucion = resolver_sistema()
            return "✅ Solución del sistema:\n" + "\n".join([f"x{i+1} = {solucion[i]:.4f}" for i in range(len(solucion))])
        except ValueError as ve:
            return f"⚠️ {ve}"
        except np.linalg.LinAlgError as e:
            return f"El sistema no tiene solución única:\n{str(e)}"
        except Exception as e:
            return f"🚫 Error inesperado:\n{str(e)}"

    @output
    @render.image
    def grafico():
        try:
            A_np, b_np, solucion = resolver_sistema()
            if A_np.shape[0] != 2:
                return None 

            fig, ax = plt.subplots()
            x_vals = np.linspace(-10, 10, 400)

            for i in range(2):
                a, b_val = A_np[i]
                c = b_np[i]
                if b_val != 0:
                    y_vals = (c - a * x_vals) / b_val
                    ax.plot(x_vals, y_vals, label=f"Ecuación {i+1}")
                else:
                    x_const = c / a
                    ax.axvline(x=x_const, label=f"Ecuación {i+1}")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Representación gráfica del sistema")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True)
            ax.legend()
            ax.plot(*solucion, 'ro') 

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            return {"src": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", "alt": "Gráfico del sistema"}

        except Exception:
            return None 

# Ejecutar la app
app = App(app_ui, server)
