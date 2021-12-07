from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
spark.conf.set("spark.streaming.stopGracefullyOnShutdown", "true")
spark.conf.set("hive.exec.max.dynamic.partitions", "3000")
spark.conf.set("hive.support.concurrency", "true")

from pyspark.sql import functions as f
from pyspark.sql import types as t



class Inference_03_obs_div():
    """Se extrae a los clientes activos en divisas, durante el periodo 
        de referencia, los cuales hayan pasado por el proceso de asignación
        de id_master, y con tipo de persona “Persona Física”. 
    """
    
    
    def __init__(self):
        self.str1="First Class"
    
    
    def export_table(self,VAR_MES):
        
        # Lectura de tablas (no es permanente)
        datos_contacto = self.from_csv_to_spark("Inputs/", "cd_cte_datos_contacto_master_202010_2608_inf.csv")
        div_cte_hist = self.from_csv_to_spark("Inputs/", "cd_div_cte_hist_202010_2608_inf.csv")
        
        
        # Secuencia de extracion de tablas
        TT_inf_pob_div =  self.poblacion(datos_contacto,div_cte_hist,VAR_MES)
        respond = TT_inf_pob_div
        return  respond
    
    
    
    # Paso 1
    def poblacion(self,datos_contacto,div_cte_hist,VAR_MES):
        
        TT_inf_pob_div = \
            div_cte_hist.alias("A").join(datos_contacto.alias("B"), f.col("A.id_master") == f.col("B.id_master"), "left") \
                .select("A.id_master", f.col("A.num_periodo_mes").alias("per_ref")) \
                .filter((f.col("A.num_periodo_mes") == VAR_MES) & \
                    (f.col("A.id_master").isNotNull()) & \
                    (f.col("A.id_master") > 0) & \
                    (f.col("A.ind_activo_div") == 1) & \
                    (f.col("B.tipo_persona") == 'FI'))
        
        return TT_inf_pob_div
    
    
    # Funciones auxs
    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf