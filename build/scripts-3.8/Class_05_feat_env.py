from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("hive.exec.dynamic.partition.mode", "nonstrict")
spark.conf.set("spark.streaming.stopGracefullyOnShutdown", "true")
spark.conf.set("hive.exec.max.dynamic.partitions", "3000")
spark.conf.set("hive.support.concurrency", "true")
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window



class Inference_05_feat_env():
    
    def __init__(self):
        self.str1="First Class"

    def export_table(self,VAR_MES,VAR_HISTORIA,TT_inf_pob_env):
        
      
        cat_fechas = self.from_csv_to_spark("Inputs/", "cd_gen_fechas_cat_202010.csv")
        cat_fechas = cat_fechas.select("num_periodo_mes").distinct()
        cd_env_cte_hist = self.from_csv_to_spark("Inputs/", "cd_env_cte_hist_202010.csv")
    
        TT_inf_feat_env = self.gen_variables_sinteticas(cat_fechas,cd_env_cte_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_env)
        respond =  TT_inf_feat_env
        return respond
    
    # paso 1: generacion de variables sinteticas de dinero express y remesas
    def gen_variables_sinteticas(self,cat_fechas,cd_env_cte_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_env):
        

        TT_inf_pob_env = \
            TT_inf_pob_env.withColumn("v_n_historia", f.to_timestamp(f.add_months(f.to_date(f.lit(str(VAR_MES +"01")),"yyyyMMdd"), - int(VAR_HISTORIA) + 1), "yyyyMM")) \
            .withColumn("historia", f.date_format(f.col("v_n_historia"), "yyyyMM").cast(t.IntegerType())) \
            .withColumn("cast", f.to_timestamp(((f.col("per_ref")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
            .drop("v_n_historia")
        
        _base_ctes = \
            TT_inf_pob_env.alias("A").join(cat_fechas.alias("B"), \
            f.col("B.num_periodo_mes").between(f.col("A.historia"), f.col("A.per_ref")),how = "left") \
            .withColumn("cast", f.to_timestamp(((f.col("A.per_ref")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
            .withColumn("diff_mes", f.months_between(f.to_date(((f.col("A.per_ref") * 100) +1).cast(t.StringType()), "yyyyMMdd"), f.col("cast")) + 1) \
            .select(f.col("A.id_master"), 
                f.col("A.per_ref"),
                f.col("A.ind_activo_rem"),
                f.col("A.ind_activo_dex"),
                f.col("B.num_periodo_mes"), 
                f.col("diff_mes").cast(t.IntegerType())) \
            .orderBy("id_master") 
          
          
          
        _env = \
            _base_ctes.alias("A").join(cd_env_cte_hist.alias("B"), (f.col("A.id_master") == f.col("B.id_master")) & \
                (f.col("A.num_periodo_mes") == f.col("B.num_periodo_mes")), how = "left") \
                .withColumn("ord_pag_dex", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_pag_dex"), f.lit(0)))))) \
                .withColumn("ord_env_dex", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_env_dex"), f.lit(0)))))) \
                .withColumn("ord_pag_rem", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_pag_rem"), f.lit(0)))))) \
                .withColumn("ord_env_rem", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_env_rem"), f.lit(0)))))) \
                .select(f.col("A.id_master"),
                    f.col("A.per_ref"),
                    f.col("A.diff_mes"),
                    f.col("A.num_periodo_mes"),
                    f.col("A.ind_activo_rem").alias("ind_activo_rem"),
                    f.abs(f.coalesce(f.col("mto_pag_rem"), f.lit(0))).alias("mto_pag_rem"),
                    f.coalesce(f.col("num_pag_rem"), f.lit(0)).alias("num_pag_rem"),
                    f.abs(f.coalesce(f.col("mto_env_rem"), f.lit(0))).alias("mto_env_rem"),
                    f.coalesce(f.col("num_env_rem"), f.lit(0)).alias("num_env_rem"),
                    f.col("A.ind_activo_dex").alias("ind_activo_dex"),
                    f.abs(f.coalesce(f.col("mto_pag_dex"), f.lit(0))).alias("mto_pag_dex"),
                    f.coalesce(f.col("num_pag_dex")).alias("num_pag_dex"),
                    f.abs(f.coalesce(f.col("mto_env_dex"), f.lit(0))).alias("mto_env_dex"),
                    f.coalesce(f.col("num_env_dex")).alias("num_env_dex"),
                    f.col("ord_pag_dex").alias("ord_pag_dex"),
                    f.col("ord_env_dex").alias("ord_env_dex"),
                    f.col("ord_pag_rem").alias("ord_pag_rem"),
                    f.col("ord_env_rem").alias("ord_env_rem")) \
                .orderBy(f.col("A.id_master"))
        
        ind_activo_dex = f.sum(f.when(f.col("diff_mes") == 1, f.col("ind_activo_dex")).otherwise(0))
        ind_activo_rem = f.sum(f.when(f.col("diff_mes") == 1, f.col("ind_activo_rem")).otherwise(0))

        ult_pag_rem = f.min(f.when(f.col("mto_pag_rem") > 0, f.col("diff_mes")).otherwise(99))
        ult_env_rem = f.min(f.when(f.col("mto_env_rem") > 0, f.col("diff_mes")).otherwise(99))
        ult_pag_dex = f.min(f.when(f.col("mto_pag_dex") > 0, f.col("diff_mes")).otherwise(99))
        ult_env_dex = f.min(f.when(f.col("mto_env_dex") > 0, f.col("diff_mes")).otherwise(99))

        num_meses_distintos_movs_rem = f.sum(f.when(f.col("num_pag_rem") + f.col("num_env_rem") > 0, 1).otherwise(0))
        num_meses_distintos_pag_rem = f.sum(f.when(f.col("num_pag_rem") > 0, 1).otherwise(0))
        num_meses_distintos_env_rem = f.sum(f.when(f.col("num_env_rem") > 0, 1).otherwise(0))
        num_meses_distintos_movs_dex = f.sum(f.when(f.col("num_pag_dex") + f.col("num_env_dex") > 0, 1).otherwise(0))
        num_meses_distintos_pag_dex = f.sum(f.when(f.col("num_pag_dex") > 0, 1).otherwise(0))
        num_meses_distintos_env_dex = f.sum(f.when(f.col("num_env_dex") > 0, 1).otherwise(0))

        tot_mto_pag_dex_corr = f.sum((f.when(f.col("ord_pag_dex").between(2, 5), f.col("mto_pag_dex")).otherwise(0) / 4) * 6)
        tot_mto_env_dex_corr = f.sum((f.when(f.col("ord_env_dex").between(2, 5), f.col("mto_env_dex")).otherwise(0) / 4) * 6)
        tot_mto_pag_rem_corr = f.sum((f.when(f.col("ord_pag_rem").between(2, 5), f.col("mto_pag_rem")).otherwise(0) / 4) * 6)
        tot_mto_env_rem_corr = f.sum((f.when(f.col("ord_env_rem").between(2, 5), f.col("mto_env_rem")).otherwise(0) / 4) * 6)

        tot_mto_pag_rem = f.sum(f.col("mto_pag_rem"))
        tot_num_pag_rem = f.sum(f.col("num_pag_rem"))
        tot_mto_env_rem = f.sum(f.col("mto_env_rem"))
        tot_num_env_rem = f.sum(f.col("num_env_rem"))
        tot_mto_pag_dex = f.sum(f.col("mto_pag_dex"))
        tot_num_pag_dex = f.sum(f.col("num_pag_dex"))
        tot_mto_env_dex = f.sum(f.col("mto_env_dex"))
        tot_num_env_dex = f.sum(f.col("num_env_dex"))

        TT_inf_feat_env = \
        _env.groupBy("id_master", "per_ref") \
            .agg(ind_activo_dex.alias("ind_activo_dex"),
                ind_activo_rem.alias("ind_activo_rem"),
                ult_pag_rem.alias("ult_pag_rem"),
                ult_env_rem.alias("ult_env_rem"),
                ult_pag_dex.alias("ult_pag_dex"),
                ult_env_dex.alias("ult_env_dex"),
                num_meses_distintos_movs_rem.alias("num_meses_distintos_movs_rem"),
                num_meses_distintos_pag_rem.alias("num_meses_distintos_pag_rem"),
                num_meses_distintos_env_rem.alias("num_meses_distintos_env_rem"),
                num_meses_distintos_movs_dex.alias("num_meses_distintos_movs_dex"),
                num_meses_distintos_pag_dex.alias("num_meses_distintos_pag_dex"),
                num_meses_distintos_env_dex.alias("num_meses_distintos_env_dex"),         
                tot_mto_pag_dex_corr.alias("tot_mto_pag_dex_corr"),
                tot_mto_env_dex_corr.alias("tot_mto_env_dex_corr"),
                tot_mto_pag_rem_corr.alias("tot_mto_pag_rem_corr"),
                tot_mto_env_rem_corr.alias("tot_mto_env_rem_corr"),         
                tot_mto_pag_rem.alias("tot_mto_pag_rem"),
                tot_num_pag_rem.alias("tot_num_pag_rem"),
                tot_mto_env_rem.alias("tot_mto_env_rem"),
                tot_num_env_rem.alias("tot_num_env_rem"),
                tot_mto_pag_dex.alias("tot_mto_pag_dex"),
                tot_num_pag_dex.alias("tot_num_pag_dex"),
                tot_mto_env_dex.alias("tot_mto_env_dex"),
                tot_num_env_dex.alias("tot_num_env_dex")) \
            .orderBy("id_master")
        
        
        return  TT_inf_feat_env
    
    
    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf