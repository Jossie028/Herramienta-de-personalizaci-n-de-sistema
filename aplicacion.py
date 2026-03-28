# Nombre: Jossie Gabriel Acosta Ruiz
# Matrícula: 24-EISN-2-029

# Importamos las librerías y modelos que usamos
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch 

# Descargamos y cargamos el Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Usar GPU si no CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Función para generar la imagen a partir del texto
def generar_imagen(prompt):
    if device == "cuda":
        with torch.amp.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=20).images[0]
    else:
        image = pipe(prompt, num_inference_steps=20).images[0]
    return image

# Creación de la interfaz visual
with gr.Blocks() as demo:
    gr.Markdown("# Generador de Fondos Inteligente")

    prompt_input = gr.Textbox(
        label="Describe tu fondo",
        placeholder="Ej: paisaje futurista azul"
    )

    # Botón de generar la imagen
    generar_btn = gr.Button("Generar")

    # Área donde se muestra la imagen
    output_image = gr.Image(label="Imagen generada")

    generar_btn.click(
        fn=generar_imagen,   # función que crea la imagen
        inputs=prompt_input, # entrada del texto
        outputs=output_image # salida de la imagen generada
    )

demo.launch()