from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set('spark.sql.parquet.compression.codec', 'snappy')
spark.conf.set('hive.exec.dynamic.partition.mode', 'nonstrict')
spark.conf.set('spark.streaming.stopGracefullyOnShutdown', 'true')
spark.conf.set('hive.exec.max.dynamic.partitions', '3000')
spark.conf.set('hive.support.concurrency', 'true')

from pyspark.sql import functions as f
from pyspark.sql.window import Window


class Modelacion_01_obs():
    
    
    def __init__(self):
        self.str1='First Class'
    
    
    def export_table(self,VAR_MES):

        
        # Lectura en server
        ingresos = spark.read.table("cd_baz_bdclientes.cd_con_cte_ingresos") \
            .select(
                f.col('id_master'),
                f.col('num_periodo_mes').alias('per_ref'),
                f.col('mto_ing_mes'),
                f.col('cod_flujo'))\
            .where(
                (f.col('num_periodo_mes') == int(VAR_MES)) & 
                (f.col('cod_flujo') == 'NOMINA') & \
                (f.col('id_master') > 0) & \
                (f.col('mto_ing_mes') > 0)) \
            .dropna(how='any') 
        
        
        
        datos_contacto = spark.read.table("cd_baz_bdclientes.cd_cte_datos_contacto_master") \
            .select(
                f.col('id_master'),
                f.col('tipo_persona'))\
            .where( (f.col('tipo_persona') == 'FI')) \
            .dropna(how='any')
        
        
        # Secuencia de extracion de tablas
        TT_train_pob_cap = self.cliente_con_renta_directa(ingresos,datos_contacto)
        return TT_train_pob_cap
    
    
    def cliente_con_renta_directa(self,ingresos,datos_contacto):    
        
        # Paso 1: Obtension de poblacion de nominados para entrenamiento    
        TT_train_aux1_cap = \
            ingresos.alias('a').join(datos_contacto.alias('b'), f.col('a.id_master') == f.col('b.id_master'), 'left')
                  
        
        # Paso 2: Generacionde numero aleatorio para obtener          
        TT_train_aux2_cap = \
            TT_train_aux1_cap.withColumn('rand', f.rand(123) * 1000) \
                .select(f.col('rand'), 'a.id_master', 'per_ref', 'mto_ing_mes')
    
    
        # Paso 3: Obtener muestra aleatoria para entrenamiento
        
        TT_train_pob_cap = \
            TT_train_aux2_cap.withColumn('rand_evt', f.row_number().over(Window.partitionBy('id_master') \
                .orderBy('rand'))) \
                .select('*') \
                .filter(f.col('rand_evt') == 1) \
                .drop('rand','rand_evt')
        
        del TT_train_aux1_cap
        del TT_train_aux2_cap
        del datos_contacto
        del ingresos
        return TT_train_pob_cap