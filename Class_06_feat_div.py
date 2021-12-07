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



class Inference_06_feat_div():
    
    def __init__(self):
        self.str1="First Class"

    def export_table(self,VAR_MES,VAR_HISTORIA,TT_inf_pob_div):
      
       
        cat_fechas = self.from_csv_to_spark("Inputs/", "cd_gen_fechas_cat_202010.csv")
        cat_fechas = cat_fechas.select("num_periodo_mes").distinct()
        cd_div_cte_hist = self.from_csv_to_spark("Inputs/", "cd_div_cte_hist_202010_2608_inf.csv")
        
        TT_inf_feat_div= self.gen_variables_sinteticas(cat_fechas,cd_div_cte_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_div)
        respond =  TT_inf_feat_div
        return respond
    
    # paso 1: generacion de variables sinteticas de divisas
    def gen_variables_sinteticas(self,cat_fechas,cd_div_cte_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_div):
        
        TT_inf_pob_div = \
            TT_inf_pob_div.withColumn("v_n_historia", f.to_timestamp(f.add_months(f.to_date(f.lit(str(VAR_MES)+"01") , "yyyyMMdd"), -int(VAR_HISTORIA) + 1), "yyyyMM")) \
           .withColumn("historia", f.date_format(f.col("v_n_historia"), "yyyyMM").cast(t.IntegerType())) \
           .withColumn("cast", f.to_timestamp(((f.col("per_ref")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
           .drop("v_n_historia")
           
        _base_ctes = \
            TT_inf_pob_div.alias("A").join(cat_fechas.alias("B"), \
                f.col("B.num_periodo_mes").between(f.col("A.historia"), f.col("A.per_ref")), how = "left") \
                .withColumn("cast", f.to_timestamp(((f.col("A.per_ref")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
                .withColumn("diff_mes", f.months_between(f.to_date(((f.col("A.per_ref") * 100) +1).cast(t.StringType()), "yyyyMMdd"), f.col("cast")) + 1) \
                .select(f.col("A.id_master"), 
                    f.col("A.per_ref"), 
                    f.col("B.num_periodo_mes"), 
                    f.col("diff_mes").cast(t.IntegerType()))


        _dvs = \
            _base_ctes.alias("A").join(cd_div_cte_hist.alias("B"), ["id_master", "num_periodo_mes"], how = "left") \
                .withColumn("ord_cpa", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_div_cpa"), f.lit(0)))))) \
                .withColumn("ord_vta", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.abs(f.coalesce(f.col("B.mto_div_vta"), f.lit(0)))))) \
                .select(f.col("A.id_master"),
                    f.col("A.per_ref"),
                    f.col("A.diff_mes"),
                    f.col("A.num_periodo_mes"),
                    f.lit(0).alias("ind_activo_div"),
                    f.abs(f.coalesce(f.col("mto_div_cpa"), f.lit(0))).alias("mto_div_cpa"),
                    f.coalesce(f.col("num_div_cpa"), f.lit(0)).alias("num_div_cpa"),
                    f.abs(f.coalesce(f.col("mto_div_vta"), f.lit(0))).alias("mto_div_vta"),
                    f.coalesce(f.col("num_div_vta"), f.lit(0)).alias("num_div_vta"),
                    f.col("ord_cpa").alias("ord_cpa"),
                    f.col("ord_vta").alias("ord_vta")) \
                .orderBy(f.col("A.id_master"))
                     
                     
        ind_activo_div = f.sum(f.when(f.col("diff_mes") == 1, f.col("ind_activo_div")).otherwise(0))

        ult_cpa_div = f.min(f.when(f.col("mto_div_cpa") > 0, f.col("diff_mes")).otherwise(99))
        ult_vta_div = f.min(f.when(f.col("mto_div_vta") > 0, f.col("diff_mes")).otherwise(99))

        num_meses_distintos_movs_div = f.sum(f.when(f.col("num_div_cpa") + f.col("num_div_vta") > 0, 1).otherwise(0))
        num_meses_distintos_cpa_div = f.sum(f.when(f.col("num_div_cpa") > 0, 1).otherwise(0))
        num_meses_distintos_vta_div = f.sum(f.when(f.col("num_div_vta") > 0, 1).otherwise(0))

        tot_mto_cpa_div_corr = f.sum((f.when(f.col("ord_cpa").between(2, 5), f.col("mto_div_cpa")).otherwise(0) / 4) * 6)
        tot_mto_vta_div_corr = f.sum((f.when(f.col("ord_vta").between(2, 5), f.col("mto_div_vta")).otherwise(0) / 4) * 6)

        tot_mto_cpa_div = f.sum(f.col("mto_div_cpa"))
        tot_num_cpa_div = f.sum(f.col("num_div_cpa"))
        tot_mto_vta_div = f.sum(f.col("mto_div_vta"))
        tot_num_vta_div = f.sum(f.col("num_div_vta"))

        TT_inf_feat_div = \
        _dvs.groupBy("id_master", "per_ref") \
            .agg(ind_activo_div.alias("ind_activo_div"),
                ult_cpa_div.alias("ult_cpa_div"),
                ult_vta_div.alias("ult_vta_div"),
                num_meses_distintos_movs_div.alias("num_meses_distintos_movs_div"),
                num_meses_distintos_cpa_div.alias("num_meses_distintos_cpa_div"),
                num_meses_distintos_vta_div.alias("num_meses_distintos_vta_div"),
                tot_mto_cpa_div_corr.alias("tot_mto_cpa_div_corr"),
                tot_mto_vta_div_corr.alias("tot_mto_vta_div_corr"),
                tot_mto_cpa_div.alias("tot_mto_cpa_div"),
                tot_num_cpa_div.alias("tot_num_cpa_div"),
                tot_mto_vta_div.alias("tot_mto_vta_div"),
                tot_num_vta_div.alias("tot_num_vta_div")) \
            .orderBy("id_master")
            
            
        return TT_inf_feat_div



    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf