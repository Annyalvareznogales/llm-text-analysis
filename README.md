# moral-values-detection
Detección de Valores Morales en Texto 

Los **valores morales** se definen como fundamentos emocionales e innatos que guían nuestras percepciones y decisiones éticas, influyendo directamente en nuestros juicios sociales. Este proyecto utiliza las dimensiones morales definidas por el Moral Foundations Theory [MFT](https://moralfoundations.org/) para clasificar textos y analizar sus implicaciones éticas, basándose en modelos preentrenados y ajustados de [Hugging Face ](https://huggingface.co/).

### Dimensiones Morales según el MFT
Los modelos implementados en este proyecto pueden identificar y clasificar textos en función de las siguientes dimensiones morales:

- **Care/Harm**: Cuidado o daño hacia los demás.
- **Fairness/Cheating**: Justicia e imparcialidad o engaño.
- **Loyalty/Betrayal**: Lealtad al grupo o traición.
- **Authority/Subversion**: Respeto o subversión de la autoridad.
- **Purity/Degradation**: Pureza o degradación moral.

Además, los modelos evalúan la **polaridad moral**, clasificando cada dimensión en dos categorías principales:
- **Virtud**: Representa aspectos positivos dentro de cada dimensión, como **Care** (cuidado) o **Fairness** (justicia).
- **Vicio**: Representa aspectos negativos dentro de cada dimensión, como **Harm** (daño) o **Cheating** (engaño).

Para desplegar los modelos, se emplea [**LitServe**](https://lightning.ai/docs/litserve/home), una herramienta especializada en convertir modelos de machine learning en servicios web. Además, se utiliza una API basada en [**FastAPI**](https://fastapi.tiangolo.com/) que interactúa con los modelos alojados a través de LitServe. FastAPI proporciona una forma sencilla de obtener el servicio web, mientras que LitServe gestiona la ejecución de los modelos y la optimización de recursos. Ambos permiten realizar predicciones de valores morales a través de solicitudes HTTP. Los modelos disponibles son:
   - [**Moral-Presence Model**](https://huggingface.co/gsi-upm/Roberta-Moral-Presence): Identifica si un texto refleja valores morales.
   - [**Moral-Polarity Model**](https://huggingface.co/gsi-upm/Roberta-Moral-Porality): Clasifica textos según la polaridad de esta moral (vicio o virtud).
   - [**Multi-Moral-Presence Model**](https://huggingface.co/gsi-upm/Roberta-MultiMoral-Presence): Determina la dimensión moral reflejada en el texto.
   - [**Multi-Moral-Polarity Model**](https://huggingface.co/gsi-upm/Roberta-MultiMoral-Polarity): Clasifica la dimensión moral distinguiendo su polaridad concreta.



## Instalación

### Requisitos Previos
- Docker y Docker Compose
- Python 3.10 o superior (si se decide ejecutar en tu entorno local)


### Configuración con Docker Compose
1. Clona este repositorio
   ```bash
   git clone https://github.com/tu_usuario/moral-values-detection.git
   cd moral-values-detection
   
2. Crea el contenedor Docker y arranca los servicios con Docker Compose
   ```bash
   docker-compose build
   docker-compose up

### Ejecución de Tests
1. En otra terminal accede al contenedor Docker del servicio Fast API
   ```bash
   docker exec -it moral-values-api /bin/bash

3. Instala httpx si no está instalado
   ```bash
   pip install httpx
   
5. Ejecuta el fichero test.py para comprobar que funciona correctamente
   ```bash
   python test.py
   
Debería reflejarse: 
 ```bash
.....
----------------------------------------------------------------------
Ran 6 tests in 0.281s

OK

```

## Ejemplos de Uso
1. Una vez levantados los contenedores, accede al contenedor Docker del servicio Fast API
   ```bash
   docker exec -it moral-values-api /bin/bash

2. Realiza una consulta a uno de los modelos disponibles utilizando *httpie* a través de la terminal, en la solicitud se debe indicar el nombre del modelo y el texto de la moral a predecir

   2.1. Instala httpie si no está instalado
      ```bash
      pip install httpie
      ```

   2.2 Consulta a **moral_model** 
      ```bash
      http POST http://localhost:8000/predict text="The government should protect its citizens and maintain law and order." model_name="moral_model"
   ```

   Respuesta esperada:
      ```bash
         HTTP/1.1 200 OK
         content-length: 100
         content-type: application/json
         date: Mon, 02 Dec 2024 09:38:54 GMT
         server: uvicorn
         
         {
             "Predicted": "MORAL",
             "Probabilities": {
                 "MORAL": 0.9998288154602051,
                 "NO-MORAL": 0.00017114622460212559
             }
         }


      ```
   
   2.3 Consulta a **moralpolarity_model**
   
      ```bash
      http POST http://localhost:8000/predict text="The government should protect its citizens and maintain law and order." model_name="moralpolarity_model" 
   ```
   
      Respuesta esperada:
      ```bash
         HTTP/1.1 200 OK
         content-length: 147
         content-type: application/json
         date: Mon, 02 Dec 2024 09:38:54 GMT
         server: uvicorn
         
         {
             "Predicted": "VIRTUE",
             "Probabilities": {
                 "NO-MORAL": 2.2218580852495506e-05,
                 "VICE": 0.00010494078014744446,
                 "VIRTUE": 0.9998729228973389
             }
         }

      ```
   
   2.4 Consulta a **multimoral_model** 
   
      ```bash
       http POST http://localhost:8000/predict text="The government should protect its citizens and maintain law and order." model_name="multimoral_model"  
   ```
   
      Respuesta esperada:
      ```bash
         HTTP/1.1 200 OK
         content-length: 293
         content-type: application/json
         date: Mon, 02 Dec 2024 09:38:54 GMT
         server: uvicorn
         
         {
             "Predicted_Moral_Trait": "AUTHORITY/SUBVERSION",
             "Probabilities": {
                 "AUTHORITY/SUBVERSION": 0.38817933201789856,
                 "CARE/HARM": 0.15376706421375275,
                 "FAIRNESS/CHEATING": 0.15128958225250244,
                 "LOYALTY/BETRAYAL": 0.17128975689411163,
                 "NO-MORAL": 0.11002018302679062,
                 "PURITY/DEGRADATION": 0.025454193353652954
             }

      ```
   
   2.5 Consulta a **multimoralpolarity_model**
   
      ```bash
       http POST http://localhost:8000/predict text="The government should protect its citizens and maintain law and order." model_name="multimoralpolarity_model"
   ```
     
      Respuesta esperada:
      ```bash
         HTTP/1.1 200 OK
         content-length: 399
         content-type: application/json
         date: Mon, 02 Dec 2024 09:38:54 GMT
         server: uvicorn
         
         {
             "Predicted_Moral": "AUTHORITY",
             "Probabilities": {
                 "AUTHORITY": 0.9737115502357483,
                 "BETRAYAL": 0.0014658872969448566,
                 "CARE": 0.0008706075605005026,
                 "CHEATING": 0.000771259656175971,
                 "DEGRADATION": 0.0009019803255796432,
                 "FAIRNESS": 0.0008387296111322939,
                 "HARM": 0.0010646103182807565,
                 "LOYALTY": 0.0004275985120330006,
                 "NO-MORAL": 0.005524156149476767,
                 "PURITY": 0.000449981598649174,
                 "SUBVERSION": 0.01397356204688549
             }
         }
   ```
