## Instalación y ejecución

```bash
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
