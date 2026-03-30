# Nombre: Jossie Gabriel Acosta Ruiz
# Matrícula: 24-EISN-2-029

# Importamos las librerías y modelos que usamos
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch 

#librerías para colores
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

# Descargamos y cargamos el Stable Diffusion ahora el 1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# optimización
pipe.enable_attention_slicing()

# Usa GPU si está disponible, si no usa CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

#función para extraer colores
def extraer_colores(image, n_colores=5):
    image = image.resize((100, 100))
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colores, n_init=10)
    kmeans.fit(pixels)

    colores = kmeans.cluster_centers_.astype(int)
    return colores

#para convertir a hex
def colores_a_hex(colores):
    return ['#%02x%02x%02x' % tuple(color) for color in colores]

# Función para generar la imagen a partir del texto
def generar_imagen(prompt):
    image = pipe(prompt, num_inference_steps=20).images[0]

    colores = extraer_colores(image)
    colores_hex = colores_a_hex(colores)

    return image, colores_hex

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

    #mostrar colores
    output_colores = gr.Textbox(label="Paleta de colores (HEX)")

    generar_btn.click(
        fn=generar_imagen,   # función que crea la imagen
        inputs=prompt_input, # entrada del texto
        outputs=[output_image, output_colores] # salida de la imagen generada
    )

demo.launch()