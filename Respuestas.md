# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

--- 

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
Añade aqui tu descripción y analisis:
> Hemos preparado el `energy_dataset.csv`, generado variables categóricas como la estación y el tipo de día y extraído estadísticos y figuras clave que permiten entender el balance del mercado pre-modelado.

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset [Hourly energy demand generation and weather] proviene de Kaggle. La variable objetivo designada es `price actual`. Tiene  sentido entrenar una regresión sobre ella porque es una variable numérica continua y el propósito fundacional del modelado de energía es predecir de forma acertada el precio del recurso, en base a la mezcla de disponibilidad y las tendencias temporales.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> La gran mayoría presenta distribuciones no-normales con sesgo a la derecha, porque muchas plantas generadoras funcionan on/off y su base generadora a menudo se centra en el cero. Al evaluar el `price actual`, se encontraron 699 outliers exactos ( 1.99% asumiendo límites superior de 96.00). Decidimos no eliminarlos dado que representan picos reales del mercado.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> De acuerdo con la matriz de correlación de Pearson, el `price actual` tiene un alto grado de correlación con:
> 1. `price day ahead` (0.732)
> 2. `generation fossil hard coal` (0.465)
> 3. `generation fossil gas` (0.461)

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> El dataset tiene pocos valores nulos marginales en las series continuas. Por ser una serie temporal preestructurada con dependencia horaria, se ha optado por propagar los últimos valores observados con una técnica Forward-fill y  Backward-fill a fin de preservar la temporalidad del precio/generación sin estropear la serie.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:
> Entrenamiento de modelo MCO Lineal validado 80-20. Además de OHE y el StandardScaler para estabilizar iteraciones y el peso predictivo frente a outliers del target.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Tras las transformaciones descritas, conseguimos:
> - **MAE**: 6.0395
> - **RMSE**: 8.6962
> - **R²**: 0.6276
> 
> El modelo funciona medianamente bien, explica cerca del 63% de la varianza total de los datos. El problema radica en que al limitarse estrictamente a una regresión lineal, le cuesta mucho adaptarse a fluctuaciones de alta volatilidad, las cuales son esperables con este tipo de datos.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:
> Ajuste a bajo nivel demostrando la mecánica OLS para cálculo y predictividad matricial.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Dicha expresión es la normalización matricial por Mínimos Cuadrados Ordinarios. La fórmula calcula explícita y matricialmente el conjunto de parámetros β que minimiza la suma de errores al cuadrado logrando la mejor proyección sobre el espacio de columnas de X. La columna de "unos" se añade para dotar de una constante "base" matemática la matriz, representando con ella el β (intercepto/sesgo).

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.864995       |
| β₁        | 2.0       | 2.063618       |
| β₂        | -1.0      | -1.117038      |
| β₃        | 0.5       | 0.438517       |

> Los coeficientes se encuentran de manera robusta muy cercanos a los reales demostrando el éxito sintético.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> He obtenido un MAE de 1.166, un RMSE de 1.461 y un R² de 0.690. Están fuertemente aproximados a las variaciones teóricas esperadas 1.2 de MAE y 1.5 de RMSE).

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> El modelo lineal sobre el dataset energético del Ejercicio 2 ha sido claramente sobrepasado por la fiabilidad sintética de Numpy, fundamentalmente porque los datos sintéticos partían de una relación 100% matemática y lineal y el nivel de ruido insertado fue bajo y de tipo Gaussiano, encajando a la perfección en un modelo matemático simple, al contrario de la imprevisibilidad subyacente y ruidosa que presenta el precio temporal del kilovatio-hora real estudiado.

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:
> Estudio de estacionariedad, perfiles residuales y temporalidad estacional prefabricada.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Sí presenta. Es una tendencia puramente lineal, creciente/positiva que arranca cerca de un volumen magnitud 50, en concordancia con la fórmula 0.05 x t + 50.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Hay estacionalidad aditiva anual con un periodo marcado en 365 días en donde existe un patrón que fluctúa repetidamente abriendo a una gran amplitud estacional periódica  de +/- 15 sumando sus curvas senoidales principales.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Perfectamente. Aisladamente, puede diferenciarse observando que la tendencia es estrictamente una "línea recta" inclinada ascendiendo constante, sobre la cual orbita lenta y orgánicamente una gigantesca onda de ciclo amortiguado cada 4 años distorsionando la recta.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Sí, estadísticamente logramos identificar un Ruido de asimilación ideal. Nuestra media extraída ha dado 0.1271 y la STD 3.22. Esto es confirmado porque al someter el nivel Residual descompuesto a un test de normalidad de Jarque-Bera, obtenemos un P-Value 0.57 de enorme superación para el alfa 0.05. Este ruido distribuido con tanta regularidad no se podría rechazar como no normal. A parte, el test estocástico ADF ratificó la inmovilidad puramente estacionaria.

---

*Fin del documento de respuestas*
