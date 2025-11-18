import os
import subprocess
import zipfile
from pyspark.sql import SparkSession

# --- Configuración de Rutas ---
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

RAW_FILE_TSV = "amazon_reviews_us_Video_Games_v1_00.tsv"
RAW_FILE_ZIP = RAW_FILE_TSV + ".zip"

RAW_PATH_TSV = os.path.join(RAW_DIR, RAW_FILE_TSV)  # Archivo descomprimido
RAW_PATH_ZIP = os.path.join(RAW_DIR, RAW_FILE_ZIP)  # Archivo descargado

PROCESSED_PATH = os.path.join(PROCESSED_DIR, "amazon_reviews_us_Video_Games.parquet")

# Crear carpetas si no existen
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -----------------------------------------------------
# PASO 1: Verificar si el archivo TSV es realmente un ZIP
# -----------------------------------------------------
def is_zip_file(filepath):
    """Verifica si un archivo es ZIP leyendo su firma mágica"""
    try:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'PK'
    except:
        return False

# Si el archivo TSV existe pero es realmente un ZIP, eliminarlo
if os.path.exists(RAW_PATH_TSV) and is_zip_file(RAW_PATH_TSV):
    print(f"⚠️  {RAW_FILE_TSV} es en realidad un ZIP. Eliminándolo...")
    os.remove(RAW_PATH_TSV)

# -----------------------------------------------------
# PASO 2: Descargar y descomprimir correctamente
# -----------------------------------------------------
if not os.path.exists(RAW_PATH_TSV):
    print(f"📥 Archivo TSV no encontrado. Iniciando proceso de descarga...")
    
    # Descargar desde Kaggle (siempre descarga como ZIP)
    print(f"🌐 Descargando desde Kaggle...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "cynthiarempel/amazon-us-customer-reviews-dataset",
            "-f", RAW_FILE_TSV,
            "-p", RAW_DIR
        ], check=True)
        print("✅ Descarga completada")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"❌ Error descargando desde Kaggle. Verifica:\n"
            f"  - Credenciales en ~/.kaggle/kaggle.json\n"
            f"  - Conexión a internet\n"
            f"  - Disponibilidad del dataset\n"
            f"Error: {e}"
        )
    
    # Kaggle descarga el archivo con extensión .zip automáticamente
    # Pero también puede descargarlo como .tsv que es un ZIP disfrazado
    downloaded_file = None
    if os.path.exists(RAW_PATH_ZIP):
        downloaded_file = RAW_PATH_ZIP
    elif os.path.exists(RAW_PATH_TSV) and is_zip_file(RAW_PATH_TSV):
        # Renombrar el archivo mal nombrado
        os.rename(RAW_PATH_TSV, RAW_PATH_ZIP)
        downloaded_file = RAW_PATH_ZIP
        print(f"🔄 Archivo renombrado a .zip")
    
    if downloaded_file:
        print(f"📦 Descomprimiendo {os.path.basename(downloaded_file)}...")
        try:
            with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            
            # Verificar que la descompresión fue exitosa
            if os.path.exists(RAW_PATH_TSV) and not is_zip_file(RAW_PATH_TSV):
                print(f"✅ Descompresión exitosa: {RAW_PATH_TSV}")
                # Eliminar el ZIP para ahorrar espacio
                os.remove(downloaded_file)
                print(f"🗑️  ZIP eliminado")
            else:
                raise FileNotFoundError(
                    f"❌ Error: {RAW_FILE_TSV} no es un archivo TSV válido después de descomprimir"
                )
        except zipfile.BadZipFile:
            raise RuntimeError(f"❌ El archivo descargado está corrupto")
    else:
        raise FileNotFoundError(f"❌ No se encontró archivo descargado para descomprimir")
else:
    print(f"✅ Archivo TSV válido encontrado: {RAW_PATH_TSV}")

# -----------------------------------------------------
# PASO 3: Leer el TSV con Spark
# -----------------------------------------------------
print(f"\n📖 Leyendo archivo TSV con Spark...")
spark = SparkSession.builder \
    .appName("Amazon Video Games Reviews") \
    .getOrCreate()

try:
    df = spark.read \
        .option("header", "true") \
        .option("sep", "\t") \
        .option("encoding", "UTF-8") \
        .option("quote", '"') \
        .option("escape", '"') \
        .option("multiLine", "true") \
        .csv(RAW_PATH_TSV)
    
    # Verificar que se leyeron datos
    row_count = df.count()
    print(f"✅ Lectura exitosa: {row_count:,} filas")
    print(f"📊 Esquema del DataFrame:")
    df.printSchema()
    
    # Mostrar muestra de datos
    print(f"\n📋 Primeras 5 filas:")
    df.show(5, truncate=True)
    
    # -----------------------------------------------------
    # PASO 4: Guardar en Parquet
    # -----------------------------------------------------
    print(f"\n💾 Guardando en formato Parquet...")
    df.write.mode("overwrite").parquet(PROCESSED_PATH)
    print(f"✅ Datos guardados en: {PROCESSED_PATH}")

except Exception as e:
    print(f"❌ Error al procesar el archivo: {e}")
    raise
finally:
    spark.stop()
    print("\n🏁 Proceso completado")