import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, trim, lower, regexp_replace, 
    to_date, year, month, dayofmonth,
    round, avg, count, lit, length, coalesce
)
from pyspark.sql.types import IntegerType, FloatType, DateType

# =====================================================
# CONFIGURACIÓN
# =====================================================

PROCESSED_DIR = "data/processed"
INPUT_PATH = os.path.join(PROCESSED_DIR, "amazon_reviews_us_Video_Games.parquet")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "amazon_reviews_cleaned.parquet")

# =====================================================
# INICIALIZAR SPARK
# =====================================================

print("🚀 Iniciando sesión de Spark...")
spark = SparkSession.builder \
    .appName("Amazon Reviews Transform") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()

# =====================================================
# 1. CARGAR DATOS
# =====================================================

print(f"\n📂 Cargando datos desde: {INPUT_PATH}")
df = spark.read.parquet(INPUT_PATH)

print(f"✅ Datos cargados: {df.count():,} filas")
print("\n📊 Esquema original:")
df.printSchema()

# =====================================================
# 2. EXPLORACIÓN INICIAL
# =====================================================

print("\n🔍 Explorando datos...")

# Contar valores nulos por columna
print("\n📉 Valores nulos por columna:")
null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in df.columns
])
null_counts.show(vertical=True)

# Mostrar muestra de datos
print("\n📋 Muestra de datos originales:")
df.show(5, truncate=True)

# =====================================================
# 3. LIMPIEZA DE DATOS
# =====================================================

print("\n🧹 Iniciando limpieza de datos...")

# 3.1 Eliminar duplicados
print("  → Eliminando duplicados...")
initial_count = df.count()
df_clean = df.dropDuplicates(['review_id'])
duplicates_removed = initial_count - df_clean.count()
print(f"    ✅ {duplicates_removed:,} duplicados eliminados")

# 3.2 Eliminar filas con valores nulos críticos
print("  → Eliminando filas con valores nulos críticos...")
initial_count = df_clean.count()
df_clean = df_clean.dropna(subset=['review_id', 'product_id', 'star_rating'])
nulls_removed = initial_count - df_clean.count()
print(f"    ✅ {nulls_removed:,} filas con nulos eliminados")

# 3.3 Limpiar texto (review_headline y review_body)
print("  → Limpiando columnas de texto...")
df_clean = df_clean \
    .withColumn('review_headline', trim(col('review_headline'))) \
    .withColumn('review_body', trim(col('review_body'))) \
    .withColumn('review_headline', 
                when(col('review_headline') == '', None)
                .otherwise(col('review_headline'))) \
    .withColumn('review_body', 
                when(col('review_body') == '', None)
                .otherwise(col('review_body')))

# =====================================================
# 4. TRANSFORMACIONES
# =====================================================

print("\n🔄 Aplicando transformaciones...")

# 4.1 Convertir tipos de datos
print("  → Convirtiendo tipos de datos...")
df_transformed = df_clean \
    .withColumn('star_rating', col('star_rating').cast(IntegerType())) \
    .withColumn('helpful_votes', col('helpful_votes').cast(IntegerType())) \
    .withColumn('total_votes', col('total_votes').cast(IntegerType()))

# 4.2 Convertir fechas (con manejo robusto de errores)
print("  → Procesando fechas...")

# Registrar UDF para parseo seguro de fechas
from pyspark.sql.functions import udf
from datetime import datetime

@udf(returnType=DateType())
def safe_parse_date(date_str):
    """Intenta parsear la fecha con múltiples formatos, devuelve None si falla"""
    if date_str is None or date_str == '':
        return None
    
    # Lista de formatos a intentar
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%m-%d-%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except:
            continue
    
    # Si ningún formato funciona, retornar None
    return None

# Aplicar la función de parseo seguro
df_transformed = df_transformed \
    .withColumn('review_date', safe_parse_date(col('review_date')))

# Crear columnas de fecha solo si review_date es válido
df_transformed = df_transformed \
    .withColumn('review_year', 
                when(col('review_date').isNotNull(), year(col('review_date')))
                .otherwise(None)) \
    .withColumn('review_month', 
                when(col('review_date').isNotNull(), month(col('review_date')))
                .otherwise(None)) \
    .withColumn('review_day', 
                when(col('review_date').isNotNull(), dayofmonth(col('review_date')))
                .otherwise(None))

# 4.3 Crear columnas derivadas
print("  → Creando columnas derivadas...")

# Calcular tasa de utilidad (helpful_votes / total_votes)
df_transformed = df_transformed.withColumn(
    'helpfulness_ratio',
    when(col('total_votes') > 0, 
         round(col('helpful_votes') / col('total_votes'), 2))
    .otherwise(None)
)

# Longitud del review
df_transformed = df_transformed.withColumn(
    'review_length',
    when(col('review_body').isNotNull(), 
         length(col('review_body')))
    .otherwise(0)
)

# Categorizar calificaciones
df_transformed = df_transformed.withColumn(
    'rating_category',
    when(col('star_rating') >= 4, 'Positive')
    .when(col('star_rating') == 3, 'Neutral')
    .otherwise('Negative')
)

# Verificado vs No verificado
df_transformed = df_transformed.withColumn(
    'is_verified',
    when(col('verified_purchase') == 'Y', True)
    .otherwise(False)
)

# =====================================================
# 5. VALIDACIONES
# =====================================================

print("\n✅ Ejecutando validaciones...")

# Validar rangos de star_rating
invalid_ratings = df_transformed.filter(
    (col('star_rating') < 1) | (col('star_rating') > 5)
).count()
print(f"  → Calificaciones inválidas: {invalid_ratings}")

# Validar fechas
invalid_dates = df_transformed.filter(col('review_date').isNull()).count()
total_dates = df_transformed.count()
print(f"  → Fechas inválidas: {invalid_dates:,} de {total_dates:,} ({invalid_dates/total_dates*100:.2f}%)")

# Validar valores negativos
negative_votes = df_transformed.filter(
    (col('helpful_votes') < 0) | (col('total_votes') < 0)
).count()
print(f"  → Votos negativos: {negative_votes}")

# =====================================================
# 6. SELECCIONAR COLUMNAS FINALES
# =====================================================

print("\n📋 Seleccionando columnas finales...")

df_final = df_transformed.select(
    # IDs
    'review_id',
    'product_id',
    'customer_id',
    
    # Información de review
    'star_rating',
    'rating_category',
    'review_headline',
    'review_body',
    'review_length',
    
    # Votos
    'helpful_votes',
    'total_votes',
    'helpfulness_ratio',
    
    # Verificación
    'verified_purchase',
    'is_verified',
    
    # Fechas
    'review_date',
    'review_year',
    'review_month',
    'review_day',
    
    # Metadata
    'marketplace',
    'product_parent',
    'product_title',
    'product_category',
    'vine'
)

# =====================================================
# 7. ESTADÍSTICAS FINALES
# =====================================================

print("\n📊 Estadísticas finales:")
print(f"  → Filas finales: {df_final.count():,}")
print(f"  → Columnas finales: {len(df_final.columns)}")

print("\n📈 Distribución de calificaciones:")
df_final.groupBy('rating_category', 'star_rating') \
    .count() \
    .orderBy('star_rating', ascending=False) \
    .show()

print("\n📅 Distribución por año:")
df_final.groupBy('review_year') \
    .agg(count('*').alias('reviews')) \
    .orderBy('review_year') \
    .show()

# =====================================================
# 8. GUARDAR DATOS LIMPIOS
# =====================================================

print(f"\n💾 Guardando datos transformados en: {OUTPUT_PATH}")
df_final.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)

print("✅ Datos guardados exitosamente")

# Mostrar muestra final
print("\n📋 Muestra de datos transformados:")
df_final.show(5, truncate=True)

print("\n📊 Esquema final:")
df_final.printSchema()

# =====================================================
# 9. CERRAR SPARK
# =====================================================

spark.stop()
print("\n🏁 Proceso de transformación completado")