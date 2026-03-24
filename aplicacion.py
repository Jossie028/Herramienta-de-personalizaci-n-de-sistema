# Nombre: Jossie Gabriel Acosta Ruiz
# Matrícula: 24-EISN-2-029

import gradio as gr


def generar_fondo(prompt):
    return "prueba"

# 🔹 Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# Generador de Fondos Inteligente")
    gr.Markdown("Escribe una descripción y genera un fondo de pantalla con colores adaptados.")

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Describe tu fondo",
            placeholder="Ej: atardecer cyberpunk morado"
        )

    generar_btn = gr.Button("Generar")

    output_text = gr.Textbox(label="Resultado")

    generar_btn.click(
        fn=generar_fondo,
        inputs=prompt_input,
        outputs=output_text
    )

# 🔹 Ejecutar app
demo.launch()