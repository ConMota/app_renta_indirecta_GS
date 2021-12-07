from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set('spark.sql.parquet.compression.codec', 'snappy')
spark.conf.set('hive.exec.dynamic.partition.mode', 'nonstrict')
spark.conf.set('spark.streaming.stopGracefullyOnShutdown', 'true')
spark.conf.set('hive.exec.max.dynamic.partitions', '3000')
spark.conf.set('hive.support.concurrency', 'true')

from pyspark.sql import functions as f



class Inference_02_obs_env():
    """Se extrae a los clientes activos en Dinero Express o Remesas, 
        durante el periodo de referencia, los cuales hayan pasado por
        el proceso de asignación de id_master, y con tipo de persona 
        “Persona Física”. 
    """
    
    def __init__(self):
        self.str1='First Class'
    
    
    def export_table(self,VAR_MES):
        
        # Lectura de tablas (no es permanente)
        datos_contacto = self.from_csv_to_spark('Inputs/', 'cd_cte_datos_contacto_master_202010_2608_inf_2.csv')
        cte_env_hist = self.from_csv_to_spark('Inputs/', 'cd_env_cte_hist_202010_2608_inf.csv')
        
        
        # Secuencia de extracion de tablas
        TT_inf_pob_env =  self.poblacion_env(datos_contacto,cte_env_hist,VAR_MES)
        respond = TT_inf_pob_env
        return  respond
    
    
    
    # Paso 1
    def poblacion_env(self,datos_contacto,cte_env_hist,VAR_MES):
        
        TT_inf_pob_env = \
            cte_env_hist.alias('A').join(datos_contacto.alias('B'), f.col('A.id_master') == f.col('B.id_master'), 'left') \
            .select('A.id_master', 
                f.col('A.num_periodo_mes').alias('per_ref'), 
                'A.ind_activo_rem',
                'A.ind_activo_dex') \
            .filter((f.col('A.num_periodo_mes') == VAR_MES) & \
                (f.col('A.id_master').isNotNull()) & \
                (f.col('A.id_master') > 0) & \
                (f.col('A.ind_activo_rem') + f.col('A.ind_activo_dex') > 0) & \
                (f.col('B.tipo_persona') == 'FI')) \
            .orderBy(f.col('A.id_master'))
        
        return TT_inf_pob_env
    
    
    # Funciones auxs
    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf