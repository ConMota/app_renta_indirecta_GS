from pyspark.sql import SparkSession
from pyspark.sql import window
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
spark.conf.set("spark.streaming.stopGracefullyOnShutdown", "true")
spark.conf.set("hive.exec.max.dynamic.partitions", "3000")
spark.conf.set("hive.support.concurrency", "true")
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window



entidades = [
    {"ID": 1, "CLAVE": 'AGU', "DESCRIPCION": 'AGUASCALIENTES'},
    {"ID": 2, "CLAVE": 'BCN', "DESCRIPCION": 'BAJA CALIFORNIA'},
    {"ID": 3, "CLAVE": 'BCS', "DESCRIPCION": 'BAJA CALIFORNIA SUR'},
    {"ID": 4, "CLAVE": 'CAM', "DESCRIPCION": 'CAMPECHE'},
    {"ID": 5, "CLAVE": 'CHP', "DESCRIPCION": 'CHIAPAS'},
    {"ID": 6, "CLAVE": 'CHH', "DESCRIPCION": 'CHIHUAHUA'},
    {"ID": 7, "CLAVE": 'CMX', "DESCRIPCION": 'CIUDAD DE MEXICO'},
    {"ID": 8, "CLAVE": 'COA', "DESCRIPCION": 'COAHUILA DE ZARAGOZA'},
    {"ID": 9, "CLAVE": 'COL', "DESCRIPCION": 'COLIMA'},
    {"ID": 10, "CLAVE": 'DUR', "DESCRIPCION": 'DURANGO'},
    {"ID": 11, "CLAVE": 'GUA', "DESCRIPCION": 'GUANAJUATO'},
    {"ID": 12, "CLAVE": 'GRO', "DESCRIPCION": 'GUERRERO'},
    {"ID": 13, "CLAVE": 'HID', "DESCRIPCION": 'HIDALGO'},
    {"ID": 14, "CLAVE": 'JAL', "DESCRIPCION": 'JALISCO'},
    {"ID": 15, "CLAVE": 'MEX', "DESCRIPCION": 'MEXICO'},
    {"ID": 16, "CLAVE": 'MIC', "DESCRIPCION": 'MICHOACAN DE OCAMPO'},
    {"ID": 17, "CLAVE": 'MOR', "DESCRIPCION": 'MORELOS'},
    {"ID": 18, "CLAVE": 'NAY', "DESCRIPCION": 'NAYARIT'},
    {"ID": 19, "CLAVE": 'NLE', "DESCRIPCION": 'NUEVO LEON'},
    {"ID": 20, "CLAVE": 'OAX', "DESCRIPCION": 'OAXACA'},
    {"ID": 21, "CLAVE": 'PUE', "DESCRIPCION": 'PUEBLA'},
    {"ID": 22, "CLAVE": 'QUE', "DESCRIPCION": 'QUERETARO'},
    {"ID": 23, "CLAVE": 'ROO', "DESCRIPCION": 'QUINTANA ROO'},
    {"ID": 24, "CLAVE": 'SLP', "DESCRIPCION": 'SAN LUIS POTOSI'},
    {"ID": 25, "CLAVE": 'SIN', "DESCRIPCION": 'SINALOA'},
    {"ID": 26, "CLAVE": 'SON', "DESCRIPCION": 'SONORA'},
    {"ID": 27, "CLAVE": 'TAB', "DESCRIPCION": 'TABASCO'},
    {"ID": 28, "CLAVE": 'TAM', "DESCRIPCION": 'TAMAULIPAS'},
    {"ID": 29, "CLAVE": 'TLA', "DESCRIPCION": 'TLAXCALA'},
    {"ID": 30, "CLAVE": 'VER', "DESCRIPCION": 'VERACRUZ'},
    {"ID": 31, "CLAVE": 'YUC', "DESCRIPCION": 'YUCATAN'},
    {"ID": 32, "CLAVE": 'ZAC', "DESCRIPCION": 'ZACATECAS'}
    ]

entidades = spark.createDataFrame(entidades)


class Inference_07_unification():
    
    def __init__(self):
        self.str1="First Class"

    def export_table(self,VAR_MES,INF_FEAT_CAP,INF_FEAT_ENV,INF_FEAT_DIV):
      
        datos_contacto = self.from_csv_to_spark("Inputs/", "cd_cte_datos_contacto_master_202010_2608.csv")
        
        TT_inf_feat_ren_ind = self.unif_customers(datos_contacto,VAR_MES,INF_FEAT_CAP,INF_FEAT_ENV,INF_FEAT_DIV)
        respond =  TT_inf_feat_ren_ind
        return respond
    
    # paso 1: unificacion de los clientes de captacion , dinero express, remesas y divisas
    def unif_customers(self,datos_contacto,VAR_MES,INF_FEAT_CAP,INF_FEAT_ENV,INF_FEAT_DIV):
        
        datos_contacto = \
            datos_contacto.withColumn("entidad", f.when(f.trim(f.col("lentidad")).isin("VERACRUZ", "VERACRUZ DE IGNACIO DE LA LLAVE"), "VERACRUZ") \
                .otherwise(f.trim(f.col("lentidad"))))
            
        _sdm = \
            datos_contacto.alias("A").join(entidades.alias("B"), f.trim(f.col("A.entidad")) == f.col("B.DESCRIPCION"), "left") \
                .withColumn("genero", f.when(f.trim(f.col("genero")).isin('N', 'E'), 'X')\
                .otherwise(f.col("genero"))) \
                .withColumn("edad", f.to_date(f.col("fecha_nacimiento"), "yyyyMMdd")) \
                .withColumn("var_mes", f.to_date(f.lit(str(VAR_MES)+"01"), "yyyyMMdd")) \
                .withColumn("edad", f.round(f.months_between(f.col("var_mes"), f.col("fecha_nacimiento")) / 12, 0).cast(t.IntegerType())) \
                .select(f.col("id_master"),
                    f.col("genero"),
                    f.col("edad"),
                    f.col("cposta").alias("cod_postal"),
                    f.coalesce(f.col("B.CLAVE"), f.lit("N/A")).alias("entidad")) \
              .orderBy("id_master")
            
            
        _union = \
            INF_FEAT_CAP.alias("A").join(INF_FEAT_DIV.alias("B"), f.col("A.id_master") == f.col("B.id_master"), "full") \
                .join(INF_FEAT_ENV.alias("C"), f.coalesce(f.col("A.id_master"), f.col("B.id_master")) == f.col("C.id_master"), "full") \
                .select(f.coalesce(f.col("A.id_master"), f.col("B.id_master"), f.col("C.id_master")).alias("id_master"),
                    f.coalesce(f.col("A.per_ref"), f.col("B.per_ref"), f.col("C.per_ref")).alias("per_ref"),
                    
                    f.coalesce(f.col("A.cod_flujo"), f.lit("NO DISPONIBLE")).alias("cod_flujo"),
                    f.coalesce(f.col("A.mto_ing_mes"), f.lit(0)).alias("mto_ing_mes"),
                    f.coalesce(f.col("A.num_flujos"), f.lit(0)).alias("num_flujos"),
                    f.coalesce(f.col("A.cod_tipo_nomina"), f.lit("NO DISPONIBLE")).alias("cod_tipo_nomina"),
                    f.coalesce(f.col("A.cod_tipo_nomina_detalle"), f.lit("NO DISPONIBLE")).alias("cod_tipo_nomina_detalle"),
                    f.coalesce(f.col("A.ind_cre_ov"), f.lit(0)).alias("ind_cre_ov"),
                    f.coalesce(f.col("A.cod_estabilidad_mto"), f.lit("NO DISPONIBLE")).alias("cod_estabilidad_mto"),
                    f.coalesce(f.col("A.cod_estabilidad_periodidad"), f.lit("NO DISPONIBLE")).alias("cod_estabilidad_periodidad"),
                    
                    f.coalesce(f.col("A.cod_perfil_trx"), f.lit("NO DISPONIBLE")).alias("cod_perfil_trx"),
                    f.coalesce(f.col("A.saldo"), f.lit(0)).alias("saldo"),
                    f.coalesce(f.col("A.potencial"), f.lit(0)).alias("potencial"),
                    f.coalesce(f.col("A.recorrido"), f.lit(0)).alias("recorrido"),
                    
                    f.coalesce(f.col("A.ind_activo_cap"), f.lit(0)).alias("ind_activo_cap"),
                    f.coalesce(f.col("A.ind_activo_afr"), f.lit(0)).alias("ind_activo_afr"),
                    f.coalesce(f.col("A.ind_activo_cre"), f.lit(0)).alias("ind_activo_cre"),
                    
                    f.coalesce(f.col("A.ult_dep_cap"), f.lit(99)).alias("ult_dep_cap"),
                    f.coalesce(f.col("A.ult_ret_cap"), f.lit(99)).alias("ult_ret_cap"),
                    f.coalesce(f.col("A.ult_dep_cte_cap"), f.lit(99)).alias("ult_dep_cte_cap"),
                    f.coalesce(f.col("A.ult_ret_cte_cap"), f.lit(99)).alias("ult_ret_cte_cap"),
                    
                    f.coalesce(f.col("A.num_meses_distintos_movs_cap"), f.lit(0)).alias("num_meses_distintos_movs_cap"),
                    f.coalesce(f.col("A.num_meses_distintos_dep_cap"), f.lit(0)).alias("num_meses_distintos_dep_cap"),
                    f.coalesce(f.col("A.num_meses_distintos_ret_cap"), f.lit(0)).alias("num_meses_distintos_ret_cap"),
                    f.coalesce(f.col("A.num_meses_distintos_movs_cte_cap"), f.lit(0)).alias("num_meses_distintos_movs_cte_cap"),
                    f.coalesce(f.col("A.num_meses_distintos_dep_cte_cap"), f.lit(0)).alias("num_meses_distintos_dep_cte_cap"),
                    f.coalesce(f.col("A.num_meses_distintos_ret_cte_cap"), f.lit(0)).alias("num_meses_distintos_ret_cte_cap"),
                    f.coalesce(f.col("A.tot_mto_dep_cap"), f.lit(0)).alias("tot_mto_dep_cap"),
                    f.coalesce(f.col("A.tot_num_dep_cap"), f.lit(0)).alias("tot_num_dep_cap"),
                    f.coalesce(f.col("A.tot_mto_ret_cap"), f.lit(0)).alias("tot_mto_ret_cap"),
                    f.coalesce(f.col("A.tot_num_ret_cap"), f.lit(0)).alias("tot_num_ret_cap"),
                    f.coalesce(f.col("A.tot_mto_dep_cte_cap"), f.lit(0)).alias("tot_mto_dep_cte_cap"),
                    f.coalesce(f.col("A.tot_num_dep_cte_cap"), f.lit(0)).alias("tot_num_dep_cte_cap"),
                    f.coalesce(f.col("A.tot_mto_ret_cte_cap"), f.lit(0)).alias("tot_mto_ret_cte_cap"),
                    f.coalesce(f.col("A.tot_num_ret_cte_cap"), f.lit(0)).alias("tot_num_ret_cte_cap"),
                    f.coalesce(f.col("A.tot_mto_dep_cap_corr"), f.lit(0)).alias("tot_mto_dep_cap_corr"),
                    f.coalesce(f.col("A.tot_mto_dep_cte_cap_corr"), f.lit(0)).alias("tot_mto_dep_cte_cap_corr"),
                    f.coalesce(f.col("B.ind_activo_div"), f.lit(0)).alias("ind_activo_div"),
                    f.coalesce(f.col("B.ult_cpa_div"), f.lit(99)).alias("ult_cpa_div"),
                    f.coalesce(f.col("B.ult_vta_div"), f.lit(99)).alias("ult_vta_div"),
                    f.coalesce(f.col("B.num_meses_distintos_movs_div"), f.lit(0)).alias("num_meses_distintos_movs_div"),
                    f.coalesce(f.col("B.num_meses_distintos_cpa_div"), f.lit(0)).alias("num_meses_distintos_cpa_div"),
                    f.coalesce(f.col("B.num_meses_distintos_vta_div"), f.lit(0)).alias("num_meses_distintos_vta_div"),
                    f.coalesce(f.col("B.tot_mto_cpa_div_corr"), f.lit(0)).alias("tot_mto_cpa_div_corr"),
                    f.coalesce(f.col("B.tot_mto_vta_div_corr"), f.lit(0)).alias("tot_mto_vta_div_corr"),
                    f.coalesce(f.col("B.tot_mto_cpa_div"), f.lit(0)).alias("tot_mto_cpa_div"),
                    f.coalesce(f.col("B.tot_num_cpa_div"), f.lit(0)).alias("tot_num_cpa_div"),
                    f.coalesce(f.col("B.tot_mto_vta_div"), f.lit(0)).alias("tot_mto_vta_div"),
                    f.coalesce(f.col("B.tot_num_vta_div"), f.lit(0)).alias("tot_num_vta_div"),
                    f.coalesce(f.col("C.ind_activo_dex"), f.lit(0)).alias("ind_activo_dex"),
                    f.coalesce(f.col("C.ind_activo_rem"), f.lit(0)).alias("ind_activo_rem"),
                    f.coalesce(f.col("C.ult_pag_rem"), f.lit(99)).alias("ult_pag_rem"),
                    f.coalesce(f.col("C.ult_env_rem"), f.lit(99)).alias("ult_env_rem"),
                    f.coalesce(f.col("C.ult_pag_dex"), f.lit(99)).alias("ult_pag_dex"),
                    f.coalesce(f.col("C.ult_env_dex"), f.lit(99)).alias("ult_env_dex"),
                    f.coalesce(f.col("C.num_meses_distintos_movs_rem"), f.lit(0)).alias("num_meses_distintos_movs_rem"),
                    f.coalesce(f.col("C.num_meses_distintos_pag_rem"), f.lit(0)).alias("num_meses_distintos_pag_rem"),
                    f.coalesce(f.col("C.num_meses_distintos_env_rem"), f.lit(0)).alias("num_meses_distintos_env_rem"),
                    f.coalesce(f.col("C.num_meses_distintos_movs_dex"), f.lit(0)).alias("num_meses_distintos_movs_dex"),
                    f.coalesce(f.col("C.num_meses_distintos_pag_dex"), f.lit(0)).alias("num_meses_distintos_pag_dex"),
                    f.coalesce(f.col("C.num_meses_distintos_env_dex"), f.lit(0)).alias("num_meses_distintos_env_dex"),
                    f.coalesce(f.col("C.tot_mto_pag_dex_corr"), f.lit(0)).alias("tot_mto_pag_dex_corr"),
                    f.coalesce(f.col("C.tot_mto_env_dex_corr"), f.lit(0)).alias("tot_mto_env_dex_corr"),
                    f.coalesce(f.col("C.tot_mto_pag_rem_corr"), f.lit(0)).alias("tot_mto_pag_rem_corr"),
                    f.coalesce(f.col("C.tot_mto_pag_rem"), f.lit(0)).alias("tot_mto_pag_rem"),
                    f.coalesce(f.col("C.tot_num_pag_rem"), f.lit(0)).alias("tot_num_pag_rem"),
                    f.coalesce(f.col("C.tot_mto_env_rem"), f.lit(0)).alias("tot_mto_env_rem"),
                    f.coalesce(f.col("C.tot_num_env_rem"), f.lit(0)).alias("tot_num_env_rem"),
                    f.coalesce(f.col("C.tot_mto_pag_dex"), f.lit(0)).alias("tot_mto_pag_dex"),
                    f.coalesce(f.col("C.tot_num_pag_dex"), f.lit(0)).alias("tot_num_pag_dex"),
                    f.coalesce(f.col("C.tot_mto_env_dex"), f.lit(0)).alias("tot_mto_env_dex"),
                    f.coalesce(f.col("C.tot_num_env_dex"), f.lit(0)).alias("tot_num_env_dex")
                    ) \
                .orderBy("A.id_master")
                       
                       
        TT_inf_feat_ren_ind = \
            _union.alias("A").join(_sdm.alias("B"), f.col("A.id_master") == f.col("B.id_master"), "left") \
                .select("A.*",
                    f.coalesce(f.col("B.genero"), f.lit("NO DISPONIBLE")).alias("genero"),
                    f.coalesce(f.col("B.edad"), f.lit(0)).alias("edad"),
                    f.coalesce(f.col("B.entidad"), f.lit("NO DISPONIBLE")).alias("entidad"),
                    f.coalesce(f.col("B.cod_postal"), f.lit(0)).alias("cod_postal")) \
                .orderBy("A.id_master")


        return TT_inf_feat_ren_ind


    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf