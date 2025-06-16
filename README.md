# API de Traductor de Lenguaje de Señas

Esta API permite reconocer gestos estáticos en lenguaje de señas a partir de imágenes.

## Estructura del Proyecto

\`\`\`
Api/
├── main.py                     # Aplicación FastAPI principal
├── models/
│   ├── predictor.py           # Lógica de predicción
│   └── saved_models/          # Modelos entrenados (.h5 y .npy)
├── utils/
│   ├── image_processing.py    # Procesamiento de imágenes
│   └── mediapipe_processor.py # Extracción de landmarks
├── schemas/
│   └── response_models.py     # Modelos de respuesta
├── requirements.txt           # Dependencias
├── Dockerfile                # Configuración Docker
└── docker-compose.yml        # Orquestación Docker
\`\`\`

## Instalación Local

1. **Clonar el repositorio y navegar a la carpeta Api:**
   \`\`\`bash
   cd Api
   \`\`\`

2. **Crear entorno virtual:**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   \`\`\`

3. **Instalar dependencias:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Colocar los modelos entrenados:**
   - Crear carpeta: `models/saved_models/`
   - Copiar los archivos:
     - `modelo_estatico_una_mano.h5`
     - `modelo_estatico_dos_manos.h5`
     - `clases_estatico_una_mano.npy`
     - `clases_estatico_dos_manos.npy`

5. **Ejecutar la API:**
   \`\`\`bash
   python main.py
   \`\`\`

La API estará disponible en: `http://localhost:8000`

## Endpoints Principales

### GET /
- **Descripción:** Endpoint de salud básico
- **Respuesta:** Estado de la API

### POST /predict
- **Descripción:** Predecir gesto desde archivo de imagen
- **Parámetros:** `file` (imagen)
- **Respuesta:** Predicción con confianza

### POST /predict_base64
- **Descripción:** Predecir gesto desde imagen en base64
- **Parámetros:** `image_base64` (string)
- **Respuesta:** Predicción con confianza

### GET /available_gestures
- **Descripción:** Obtener lista de gestos disponibles
- **Respuesta:** Lista de gestos por categoría

### GET /health
- **Descripción:** Estado detallado de la API y modelos
- **Respuesta:** Estado de carga de modelos

## Documentación Interactiva

Una vez ejecutando la API, puedes acceder a:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Deployment

### Opción 1: Docker Local
\`\`\`bash
docker build -t gesture-api .
docker run -p 8000:8000 -v $(pwd)/models/saved_models:/app/models/saved_models gesture-api
\`\`\`

### Opción 2: Docker Compose
\`\`\`bash
docker-compose up -d
\`\`\`

### Opción 3: Servicios en la Nube

#### Railway (Recomendado - Fácil y Gratuito)
1. Crear cuenta en [Railway](https://railway.app)
2. Conectar repositorio GitHub
3. Configurar variables de entorno si es necesario
4. Deploy automático

#### Render
1. Crear cuenta en [Render](https://render.com)
2. Conectar repositorio GitHub
3. Configurar como Web Service
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Heroku
1. Instalar Heroku CLI
2. Crear `Procfile`:
   \`\`\`
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   \`\`\`
3. Deploy:
   \`\`\`bash
   heroku create tu-app-name
   git push heroku main
   \`\`\`

## Uso desde Flutter

### Ejemplo de petición POST con archivo:
\`\`\`dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<Map<String, dynamic>> predictGesture(File imageFile) async {
  var request = http.MultipartRequest(
    'POST', 
    Uri.parse('https://tu-api-url.com/predict')
  );
  
  request.files.add(
    await http.MultipartFile.fromPath('file', imageFile.path)
  );
  
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  
  return json.decode(responseData);
}
\`\`\`

### Ejemplo de petición POST con base64:
\`\`\`dart
Future<Map<String, dynamic>> predictGestureBase64(String base64Image) async {
  final response = await http.post(
    Uri.parse('https://tu-api-url.com/predict_base64'),
    headers: {'Content-Type': 'application/json'},
    body: json.encode({'image_base64': base64Image}),
  );
  
  return json.decode(response.body);
}
\`\`\`

## Configuración de Producción

Para producción, considera:

1. **Variables de entorno:**
   - `CONFIDENCE_THRESHOLD`: Umbral de confianza
   - `MAX_IMAGE_SIZE`: Tamaño máximo de imagen
   - `LOG_LEVEL`: Nivel de logging

2. **Optimizaciones:**
   - Usar gunicorn con múltiples workers
   - Implementar cache para modelos
   - Configurar límites de rate limiting

3. **Seguridad:**
   - Implementar autenticación si es necesario
   - Configurar CORS apropiadamente
   - Validar tamaños de archivo

## Troubleshooting

### Error: "No se encontraron modelos"
- Verificar que los archivos .h5 y .npy están en `models/saved_models/`
- Verificar permisos de lectura

### Error: "MediaPipe no funciona"
- Instalar dependencias del sistema (ver Dockerfile)
- Verificar versión de OpenCV compatible

### Error de memoria
- Reducir tamaño de imágenes de entrada
- Configurar límites de memoria en Docker
