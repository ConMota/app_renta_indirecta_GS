from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set('spark.sql.parquet.compression.codec', 'snappy')
spark.conf.set('hive.exec.dynamic.partition.mode', 'nonstrict')
spark.conf.set('spark.streaming.stopGracefullyOnShutdown', 'true')
spark.conf.set('hive.exec.max.dynamic.partitions', '3000')
spark.conf.set('hive.support.concurrency', 'true')

from pyspark.sql import functions as f
from pyspark.sql import types as t

# variables globales

class Modelacion_02_feat():
    
    
    def __init__(self):
        self.str1='First Class'
    
    
    def export_table(self,TRAIN_POB_CAP,VAR_MES):

        # Lectura en el server
        datos_contacto = spark.read.table("cd_baz_bdclientes.cd_cte_datos_contacto_master") \
            .select(
                f.col('id_master'),
                f.col('lentidad'),
                f.col('genero'),
                f.col('fecha_nacimiento'),
                f.col('cposta').alias('cod_postal')) \
            .withColumn('entidad', f.when(f.trim(f.col('lentidad')).isin('VERACRUZ', 'VERACRUZ DE IGNACIO DE LA LLAVE'), 'VERACRUZ') \
            .otherwise(f.trim(f.col('lentidad')))) \
            .drop(f.col('lentidad'))
        
        
        recorrido = spark.read.table("cd_baz_bdclientes.cd_con_cte_recorrido") \
            .select(
                f.col('id_master'),
                f.col('num_periodo_mes').alias('per_ref'),
                f.col('cod_perfil_trx'),
                f.col('saldo'),
                f.col('potencial'),
                f.col('recorrido')) \
            .filter(f.col('per_ref') == str(VAR_MES)) \
            .orderBy(f.col('id_master'))

        # Secuencia de extracion de tablas
        TT_train_feat_ren_ind =  self.feat_cap(recorrido,datos_contacto,VAR_MES,TRAIN_POB_CAP)
        respond = TT_train_feat_ren_ind
        return  respond
    
    
    
    # Paso 1: Extraccion de informacion para el modelo de potenciales
    def feat_cap(self,recorrido,datos_contacto,VAR_MES,TRAIN_POB_CAP):
       
        
        _sdm = \
            datos_contacto.alias('A').withColumn('genero', f.when(f.trim(f.col('genero')).isin('N', 'E'), 'X') \
                .otherwise(f.col('genero'))) \
                .withColumn('var_mes', f.to_date(f.lit(str(VAR_MES)+'01'), 'yyyyMMdd')) \
                .withColumn('edad', f.round(f.months_between(f.col('var_mes'), f.col('fecha_nacimiento')) / 12, 0).cast(t.IntegerType())) \
                .select(
                    f.col('id_master'),
                    f.col('edad'),
                    f.col('var_mes'),
                    f.col('genero'),
                    f.col('cod_postal'),
                    f.col('entidad')) \
                .orderBy('id_master')
        
        
        TT_train_feat_ren_ind = \
            TRAIN_POB_CAP.alias('A').join(_sdm.alias('B'), f.col('A.id_master') == f.col('B.id_master'), 'left') \
                .join(recorrido.alias('D'), f.col('A.id_master') == f.col('D.id_master'), 'left') \
                .select(
                    f.col('A.id_master'),
                    f.col('A.per_ref'),
                    f.col('A.mto_ing_mes'),                         
                    f.coalesce(f.col('B.genero'), f.lit('VACIO')).alias('genero'),
                    f.coalesce(f.col('B.edad'), f.lit(0)).alias('edad'), # mayor a 18
                    f.coalesce(f.col('B.entidad'), f.lit('VACIO')).alias('entidad'),
                    f.coalesce(f.col('B.cod_postal'), f.lit(0)).alias('cod_postal'),
                    f.coalesce(f.col('D.saldo'), f.lit(0)).alias('saldo'),
                    f.coalesce(f.col('D.potencial'), f.lit(0)).alias('potencial'),
                    f.coalesce(f.col('D.recorrido'), f.lit(0)).alias('recorrido')) \
                .orderBy('id_master')
        
        
        del datos_contacto
        del recorrido
        del _sdm

        return TT_train_feat_ren_ind
    
