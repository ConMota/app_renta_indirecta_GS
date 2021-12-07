from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set('spark.sql.parquet.compression.codec', 'snappy')
spark.conf.set('hive.exec.dynamic.partition.mode', 'nonstrict')
spark.conf.set('spark.streaming.stopGracefullyOnShutdown', 'true')
spark.conf.set('hive.exec.max.dynamic.partitions', '3000')
spark.conf.set('hive.support.concurrency', 'true')
from pyspark.sql.window import Window
from pyspark.sql import functions as f
from pyspark.sql import types as t


class Inference_01_obs_cap():
    
    """Se extrae a los clientes activos en captación, durante el periodo de referencia, los cuales hayan pasado 
        por el proceso de asignación de id_master, y con tipo de persona “Persona Física”; adicionalmente se 
        extrae a los clientes a los que se les calculó renta directa durante el mismo periodo, dentro de los 
        cuales se encuentran clientes de captación, afore y crédito. 
    """
    
    def __init__(self):
        self.str1='First Class'
    
    
    def export_table(self,VAR_MES):
        
        # Lectura de tablas (no es permanente)
        path_files_csv = 'Inputs/' 
        cap_cte = self.from_csv_to_spark(path_files_csv, 'cd_cap_cte_hist.csv')
        ingresos = self.from_csv_to_spark(path_files_csv, 'cd_con_cte_ingresos_202010_2608_inf.csv')

        
        # Secuencia de extracion de tablas
        TT_inf_pob_cap =  self.poblacion_cap(cap_cte,ingresos,VAR_MES)
        respond = TT_inf_pob_cap
        return  respond
    
    
    
    # Paso 1
    def poblacion_cap(self,cap_cte,ingresos,var_mes):
        
        _cap = \
            cap_cte.withColumn('prio', f.dense_rank().over(Window.partitionBy(['id_master', 'num_periodo_mes']) \
                .orderBy(f.desc('ind_activo_cap'), f.desc('sld_tot_prod_cap')))) \
                .select('id_master', 'ind_activo_cap', 'num_periodo_mes') \
                .filter((f.col('num_periodo_mes') == var_mes) & \
                    (f.col('id_master').isNotNull()) & \
                    (f.col('id_master') > 0) & \
                    (f.col('ind_activo_cap') == 1) & \
                    (f.col('cod_tipo_persona') == 'PF') & \
                    (f.col('prio') == 1)) \
                .orderBy('id_master')

        _ren = \
            ingresos.select('id_master', 'num_periodo_mes', 
                    f.coalesce(f.col('ind_cre'), f.lit(0)).alias('ind_activo_cre'),
                    f.coalesce(f.col('ind_afr'), f.lit(0)).cast(t.IntegerType()).alias('ind_activo_afr')) \
                .filter((f.col('num_periodo_mes') == var_mes) & \
                (f.col('id_master') > 0)) \
                .orderBy('id_master')
        
        TT_inf_pob_cap = \
            _cap.alias('A').join(_ren.alias('B'), f.col('A.id_master') == f.col('B.id_master'), 'full') \
               .select(f.coalesce(f.col('A.id_master'), f.col('B.id_master')).alias('id_master'),
                    f.coalesce(f.col('A.num_periodo_mes'), f.col('B.num_periodo_mes')).alias('per_ref'),
                    f.coalesce(f.col('A.ind_activo_cap'), f.lit(0)).alias('ind_activo_cap'),
                    f.coalesce(f.col('B.ind_activo_cre'), f.lit(0)).alias('ind_activo_cre'),
                    f.coalesce(f.col('B.ind_activo_afr'), f.lit(0)).alias('ind_activo_afr')) \
               .orderBy('id_master', 'per_ref')
        
        
        return TT_inf_pob_cap
    
    
    # Funciones auxs
    def from_csv_to_spark(self,path_files_csv, file):
        sdf = spark.read.csv(path_files_csv + file, inferSchema = True, header = True)
        return sdf