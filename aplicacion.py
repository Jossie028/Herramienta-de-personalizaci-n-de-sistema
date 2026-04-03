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

#optimización
pipe.enable_attention_slicing()

# Usa GPU si está disponible, si no usa CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


# Funcion para presentar cuadros de colores
def generar_html_colores(colores_hex):
    html = ""
    for color in colores_hex:
        html += f"""
        <div style="
            width: 60px;
            height: 60px;
            background-color: {color};
            display: inline-block;
            margin: 5px;
            border-radius: 8px;
            border: 1px solid #000;
        "></div>
        """
    return html


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

    html_colores = generar_html_colores(colores_hex)

    principal = colores_hex[0]
    secundario = colores_hex[1]
    texto = color_texto(colores)

    css = generar_css(principal, secundario, texto)

    return image, colores_hex, html_colores, css


#Obtener el color ptincipal
def color_principal(colores):
    return colores[0] 

#color de textos dinamicos
def color_texto(colores):
    promedio = np.mean(colores)
    return "white" if promedio < 128 else "black"


#Creamos el estilo dinamico
def generar_css(color_principal, color_secundario, color_texto):
    return f"""
    <style>
        .gradio-container {{
            background: linear-gradient(135deg, {color_principal}, {color_secundario}) !important;
            color: {color_texto} !important;
        }}

        h1 {{
            background: {color_secundario};
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }}

        .gr-box, .gr-panel {{
            background-color: rgba(255,255,255,0.1) !important;
            border: 2px solid {color_secundario} !important;
            border-radius: 12px;
        }}

        button {{
            background-color: {color_secundario} !important;
            color: {color_texto} !important;
            border-radius: 10px;
            border: none;
        }}

        .gr-image {{
            border: 4px solid {color_secundario};
            border-radius: 12px;
        }}

        textarea, input {{
            border: 2px solid {color_secundario} !important;
            border-radius: 8px;
        }}
    </style>
    """


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

    output_html = gr.HTML(label="Vista de colores")

    output_style = gr.HTML()

    generar_btn.click(
        fn=generar_imagen,
        inputs=prompt_input,
        outputs=[output_image, output_colores, output_html, output_style]
    )

demo.launch(inline=True)