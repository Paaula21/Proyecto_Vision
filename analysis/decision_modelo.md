# Modelo elegido para detectar frutas

## ¿Qué tenemos que hacer?

El objetivo es encontrar las frutas que aparecen en cada imagen. Para cada una, el programa tiene que indicar su especie y marcar los píxeles que le pertenecen. También debe dar una puntuación de confianza.

Esto se llama **segmentación por instancias**: si aparecen dos manzanas juntas, necesitamos una máscara distinta para cada una, no una única zona que marque todas las manzanas.

## ¿Por qué he elegido Mask R-CNN?

He elegido **Mask R-CNN** como primer modelo porque puede detectar cada fruta, identificar su especie y generar una máscara en el mismo proceso. Además, podemos partir de un modelo preentrenado y adaptarlo a nuestras imágenes, en lugar de empezar desde cero.

El modelo utiliza una red llamada **ResNet-50** para extraer características de la imagen y combina información de distintos tamaños mediante **FPN**. Esto puede ser útil porque las frutas aparecen a diferentes distancias y tamaños, aunque tendremos que comprobar cómo funciona con las más pequeñas.

## ¿Cómo funciona?

El proceso es:

1. El modelo analiza la imagen y busca zonas donde puede haber objetos.
2. Para cada zona, intenta reconocer la especie de fruta.
3. Genera una máscara con los píxeles que considera parte de esa fruta.
4. Asigna una puntuación que indica la confianza de la detección.

Las categorías que utilizamos son naranja, manzana, pera y limón. El modelo también tiene una categoría de fondo para lo que no es fruta.

## ¿Qué hemos preparado?

Hasta ahora hemos preparado tres archivos:

- `src/model.py`: crea el modelo y lo adapta a nuestras categorías.
- `src/dataset.py`: lee las imágenes, las máscaras y las anotaciones, y comprueba la división de los datos.
- `train.py`: utiliza los datos de entrenamiento para ajustar el modelo y guardar sus pesos.

El código incluye las anotaciones marcadas como `iscrowd` durante el entrenamiento. Son frutas muy pequeñas que el evaluador ignora. Es una decisión inicial que tendremos que revisar cuando podamos medir los resultados.

Todavía no tenemos unos pesos entrenados y evaluados, así que no podemos afirmar que el modelo funcione bien ni que sea mejor que otras opciones.

## Comparación con otras técnicas

- **Umbrales de color y regiones:** son más sencillos de entender y se parecen al ejemplo del proyecto. Pueden fallar cuando la fruta tiene un color parecido a las hojas o cuando varias frutas están juntas.
- **K-means y GMM:** agrupan píxeles con características parecidas. Pueden ayudar a separar colores, pero después habría que decidir qué zonas son frutas y distinguir unas frutas de otras.
- **Descriptores de forma y color:** ayudan a describir y clasificar una región, pero primero necesitamos localizar esa región en la imagen.
- **U-Net:** puede crear una imagen que indique qué píxeles son fruta, pero necesitaría un paso adicional para separar frutas de la misma especie.
- **SAM:** genera máscaras y lo hemos visto en las prácticas. Por sí solo no indica si una máscara corresponde a una naranja, una manzana, una pera o un limón.
- **CLIP:** puede ayudar a reconocer el contenido de una imagen o de un recorte, pero por sí solo no localiza cada fruta ni produce su máscara.

He elegido Mask R-CNN porque reúne detección, clasificación y generación de máscaras. A cambio, es más complejo de entrenar y necesita más recursos que los métodos clásicos. No hemos comparado todavía sus resultados, por lo que esta elección es un punto de partida.

## Limitaciones y siguientes pasos

Muchas frutas son pequeñas o están parcialmente tapadas por hojas. Esto puede dificultar que el modelo las detecte y dibuje bien sus máscaras.

Además, cada división deja fuera un árbol y, con él, una especie completa. Por eso tendremos que explicar con cuidado qué podemos medir con esa validación. Sirve para estudiar la detección en un árbol que no se ha usado en ese entrenamiento, pero no para comprobar normalmente el reconocimiento de una especie que ha quedado fuera.

El siguiente paso es completar una prueba de entrenamiento, generar predicciones y evaluarlas. También será útil compararlas con un método más sencillo para ver si el modelo aporta una mejora.
