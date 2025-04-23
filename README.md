## Instalación y ejecución

```bash
# ⚠️DESCARGAR EL ARCHIVO DEL MODELO Y GUARDARLO EN LA CARPETA MODEL⚠️
https://drive.google.com/file/d/1pNpuXauJL0F-h-OFNHJJvDBqF5i0k0N0/view?usp=sharing

# Clonar el repositorio
git clone https://github.com/alexisloor/analisis-mamografias.git
cd analisis-mamografias

# Crear y activar entorno virtual
python -m venv venv
source venv\Scripts\activate

# Instalar librerías
pip install -r requirements.txt

# Ejecutar la aplicación
python -m uvicorn main:app --reload
