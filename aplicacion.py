# Nombre: Jossie Gabriel Acosta Ruiz
# Matrícula: 24-EISN-2-029

#Importamos las liberias y modelos que usamos
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch 


# descargamos y cargamos el Stable DIfussion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# Por si no tenemos grafica
pipe = pipe.to("cpu")

# Función para generar la imagen a partir del texto
def generar_imagen(prompt):
    image = pipe(prompt).images[0]
    return image

#Creacion de la interfaz visual
import gradio as gr

def generar_imagen(prompt):
    image = pipe(prompt).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("#Generador de Fondos Inteligente")

    prompt_input = gr.Textbox(
        label="Describe tu fondo",
        placeholder="Ej: paisaje futurista azul"
    )

#boton de generar la imagen
    generar_btn = gr.Button("Generar")

#area donde se muestra la imagen
    output_image = gr.Image(label="Imagen generada")

    generar_btn.click(
        fn=generar_imagen, #funcion que crea la imagen
        inputs=prompt_input,#entrada del texto
        outputs=output_image #salida de la imagen generada
    )

demo.launch()

