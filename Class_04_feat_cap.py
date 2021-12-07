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



class Inference_04_feat_cap():
    
    def __init__(self):
        self.str1="First Class"

    def export_table(self,VAR_MES,VAR_HISTORIA,TT_inf_pob_cap):
        
        cap_cuenta_hist = self.from_csv_to_spark("Inputs/", "cd_cap_cuenta_hist_2608_full.csv")
        cd_con_cte_ingresos = self.from_csv_to_spark("Inputs/", "cd_con_cte_ingresos_202010_2608_inf.csv")
        cd_con_cte_recorrido = self.from_csv_to_spark("Inputs/", "cd_con_cte_recorrido_202010_2608.csv")
        
        id_master = list(TT_inf_pob_cap.select(f.col('id_master')).toPandas()['id_master'])

        TT_inf_aux1_cap = self.info_captacion(cap_cuenta_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_cap,id_master)
        TT_inf_aux2_cap = self.gen_var_sinteticas(TT_inf_pob_cap,TT_inf_aux1_cap)
        TT_inf_feat_cap = self.unif_inf_pob(cd_con_cte_ingresos,cd_con_cte_recorrido,TT_inf_aux2_cap)
        respond = TT_inf_feat_cap
        return respond

    # paso 1: obtencion de informacion a nivel cuenta,mensual de los clientes de captacion
    def info_captacion(self,cap_cuenta_hist,VAR_MES,VAR_HISTORIA,TT_inf_pob_cap,id_master):

        TT_inf_aux1_cap = \
            cap_cuenta_hist.withColumn("v_n_historia", f.to_timestamp(f.add_months(f.to_date(f.lit(str(VAR_MES)+"01") , "yyyyMMdd"), - int(VAR_HISTORIA) + 1), "yyyyMM")) \
                .withColumn("historia", f.date_format(f.col("v_n_historia"), "yyyyMM").cast(t.IntegerType())) \
                .withColumn("cast", f.to_timestamp(((f.col("num_periodo_mes")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
                .withColumn("diff_mes", f.months_between(f.to_date(f.lit(str(VAR_MES)+"01"), "yyyyMMdd"), f.col("cast")) + 1) \
                .select(f.col("id_master"),
                    f.lit(VAR_MES).alias("per_ref"),
                    f.col("num_periodo_mes"),
                    f.col("diff_mes").cast(t.IntegerType()),
                    (f.col("mto_dep_transferencia") + 
                    f.col("mto_dep_cheque") + 
                    f.col("mto_dep_efectivo") + 
                    f.col("mto_dep_nomina") + 
                    f.col("mto_dep_traspaso_terceros")).alias("mto_dep_cap"),
                    (f.col("num_dep_transferencia") + 
                    f.col("num_dep_cheque") + 
                    f.col("num_dep_efectivo") + 
                    f.col("num_dep_nomina") + 
                    f.col("num_dep_traspaso_terceros")).alias("num_dep_cap"),
                    (f.col("mto_ret_transferencia") + 
                    f.col("mto_ret_cheque") + 
                    f.col("mto_ret_efectivo") + 
                    f.col("mto_ret_nomina") + 
                    f.col("mto_ret_traspaso_terceros")).alias("mto_ret_cap"),
                    (f.col("num_ret_transferencia") + 
                    f.col("num_ret_cheque") + 
                    f.col("num_ret_efectivo") + 
                    f.col("num_ret_nomina") + 
                    f.col("num_ret_traspaso_terceros")).alias("num_ret_cap"),
                    (f.col("mto_dep_transferencia") + 
                    f.col("mto_dep_cheque") + 
                    f.col("mto_dep_efectivo") + 
                    f.col("mto_dep_traspaso_terceros")).alias("mto_dep_cte_cap"),
                    (f.col("num_dep_transferencia") + 
                    f.col("num_dep_cheque") + 
                    f.col("num_dep_efectivo") + 
                    f.col("num_dep_traspaso_terceros")).alias("num_dep_cte_cap"),
                    (f.col("mto_ret_transferencia") + 
                    f.col("mto_ret_cheque") + 
                    f.col("mto_ret_efectivo") + 
                    f.col("mto_ret_traspaso_terceros")).alias("mto_ret_cte_cap"),
                    (f.col("num_ret_transferencia") + 
                    f.col("num_ret_cheque") + 
                    f.col("num_ret_efectivo") + 
                    f.col("num_ret_traspaso_terceros")).alias("num_ret_cte_cap")) \
               .filter((f.col("num_periodo_mes").between(f.col("historia"), VAR_MES)) & \
                       (f.col("id_master").isin(id_master))) \
               .dropDuplicates() \
               .orderBy(f.col("id_master"))
        
        TT_inf_aux1_cap = \
            cap_cuenta_hist.alias("A").join(TT_inf_pob_cap.alias("B"), f.col("A.id_master") == f.col("B.id_master"), how = "inner") \
                .withColumn("v_n_historia", f.to_timestamp(f.add_months(f.to_date(f.lit(str(VAR_MES)+"01") \
                , "yyyyMMdd"), -int(VAR_HISTORIA) + 1), "yyyyMM")) \
                    .withColumn("historia", f.date_format(f.col("v_n_historia"), "yyyyMM").cast(t.IntegerType())) \
                    .withColumn("cast", f.to_timestamp(((f.col("A.num_periodo_mes")*100) + 1).cast(t.StringType()), "yyyyMMdd")) \
                    .withColumn("diff_mes", f.months_between(f.to_date(f.lit(str(VAR_MES)+"01"), "yyyyMMdd"), f.col("cast")) + 1) \
                    .select(f.col("A.id_master"),
                        f.lit(VAR_MES).alias("per_ref"),
                        f.col("A.num_periodo_mes"),
                        f.col("diff_mes").cast(t.IntegerType()),
                        (f.col("mto_dep_transferencia") + 
                        f.col("mto_dep_cheque") + 
                        f.col("mto_dep_efectivo") + 
                        f.col("mto_dep_nomina") + 
                        f.col("mto_dep_traspaso_terceros")).alias("mto_dep_cap"),
                        (f.col("num_dep_transferencia") + 
                        f.col("num_dep_cheque") + 
                        f.col("num_dep_efectivo") + 
                        f.col("num_dep_nomina") + 
                        f.col("num_dep_traspaso_terceros")).alias("num_dep_cap"),
                        (f.col("mto_ret_transferencia") + 
                        f.col("mto_ret_cheque") + 
                        f.col("mto_ret_efectivo") + 
                        f.col("mto_ret_nomina") + 
                        f.col("mto_ret_traspaso_terceros")).alias("mto_ret_cap"),
                        (f.col("num_ret_transferencia") + 
                        f.col("num_ret_cheque") + 
                        f.col("num_ret_efectivo") + 
                        f.col("num_ret_nomina") + 
                        f.col("num_ret_traspaso_terceros")).alias("num_ret_cap"),
                        (f.col("mto_dep_transferencia") + 
                        f.col("mto_dep_cheque") + 
                        f.col("mto_dep_efectivo") + 
                        f.col("mto_dep_traspaso_terceros")).alias("mto_dep_cte_cap"),
                        (f.col("num_dep_transferencia") + 
                        f.col("num_dep_cheque") + 
                        f.col("num_dep_efectivo") + 
                        f.col("num_dep_traspaso_terceros")).alias("num_dep_cte_cap"),
                        (f.col("mto_ret_transferencia") + 
                        f.col("mto_ret_cheque") + 
                        f.col("mto_ret_efectivo") + 
                        f.col("mto_ret_traspaso_terceros")).alias("mto_ret_cte_cap"),
                        (f.col("num_ret_transferencia") + 
                        f.col("num_ret_cheque") + 
                        f.col("num_ret_efectivo") + 
                        f.col("num_ret_traspaso_terceros")).alias("num_ret_cte_cap")) \
                    .filter(f.col("A.num_periodo_mes").between(f.col("historia"), VAR_MES)) \
                    .dropDuplicates() \
                    .orderBy(f.col("A.id_master"))
        
        return TT_inf_aux1_cap

    # paso 2: generacion de variables sinteticas de captacion
    def gen_var_sinteticas(self,TT_inf_pob_cap,TT_inf_aux1_cap):
        
        mto_dep_cap = f.when(f.abs(f.sum("mto_dep_cap")) < 100, 0).otherwise(f.abs(f.sum("mto_dep_cap")))
        num_dep_cap = f.when(f.abs(f.sum("num_dep_cap")) < 100, 0).otherwise(f.abs(f.sum("num_dep_cap")))
        mto_ret_cap = f.abs(f.sum("mto_ret_cap"))
        num_ret_cap = f.sum("num_ret_cap")
        mto_dep_cte_cap = f.when(f.abs(f.sum("mto_dep_cte_cap")) < 100, 0).otherwise(f.abs(f.sum("mto_dep_cte_cap")))
        num_dep_cte_cap = f.when(f.abs(f.sum("num_dep_cte_cap")) < 100, 0).otherwise(f.abs(f.sum("num_dep_cte_cap")))
        mto_ret_cte_cap = f.abs(f.sum("mto_ret_cte_cap")) 
        num_ret_cte_cap = f.sum("num_ret_cte_cap")

        _agg_cta = \
        TT_inf_aux1_cap.groupBy(["id_master", "num_periodo_mes", "per_ref", "diff_mes"]) \
            .agg(mto_dep_cap.alias("mto_dep_cap"),
                num_dep_cap.alias("num_dep_cap"),
                mto_ret_cap.alias("mto_ret_cap"),
                num_ret_cap.alias("num_ret_cap"),
                mto_dep_cte_cap.alias("mto_dep_cte_cap"),
                num_dep_cte_cap.alias("num_dep_cte_cap"),
                mto_ret_cte_cap.alias("mto_ret_cte_cap"),
                num_ret_cte_cap.alias("num_ret_cte_cap")) \
                .orderBy("id_master")
        
        _cap = \
            TT_inf_pob_cap.alias("A").join(_agg_cta.alias("B"), f.col("A.id_master") == f.col("B.id_master"), how = "inner") \
                .withColumn("ord_nom", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.col("B.mto_dep_cap")))) \
                .withColumn("ord_cte", f.row_number().over(Window.partitionBy(f.col("A.id_master")) \
                .orderBy(f.col("B.mto_dep_cte_cap")))) \
                .select(f.col("A.id_master"),
                f.col("A.per_ref"),
                f.col("num_periodo_mes"),
                f.col("A.ind_activo_cap"),
                f.col("A.ind_activo_afr"),
                f.col("A.ind_activo_cre"),
                f.col("diff_mes"),
                f.col("mto_dep_cap"),
                f.col("num_dep_cap"),
                f.col("mto_ret_cap"),
                f.col("num_ret_cap"),
                f.col("mto_dep_cte_cap"),
                f.col("num_dep_cte_cap"),
                f.col("mto_ret_cte_cap"),
                f.col("num_ret_cte_cap"),
                f.col("ord_nom"),
                f.col("ord_cte")) \
                .orderBy(f.col("A.id_master"))
        
        
        ind_activo_cap = f.max("ind_activo_cap")
        ind_activo_afr = f.max("ind_activo_afr")
        ind_activo_cre = f.max("ind_activo_cre")

        ult_dep_cap = f.min(f.when(f.col("mto_dep_cap") > 0, f.col("diff_mes")).otherwise(99))
        ult_ret_cap = f.min(f.when(f.col("mto_ret_cap") > 0, f.col("diff_mes")).otherwise(99))
        ult_dep_cte_cap = f.min(f.when(f.col("mto_dep_cte_cap") > 0, f.col("diff_mes")).otherwise(99))
        ult_ret_cte_cap = f.min(f.when(f.col("mto_ret_cte_cap") > 0, f.col("diff_mes")).otherwise(99))

        num_meses_distintos_movs_cap = f.sum(f.when(f.col("num_dep_cap") + f.col("num_ret_cap") > 0, 1).otherwise(0))
        num_meses_distintos_dep_cap = f.sum(f.when(f.col("num_dep_cap") > 0, 1).otherwise(0))
        num_meses_distintos_ret_cap = f.sum(f.when(f.col("num_ret_cap") > 0, 1).otherwise(0))
        num_meses_distintos_movs_cte_cap = f.sum(f.when(f.col("num_dep_cte_cap") + f.col("num_ret_cte_cap") > 0, 1).otherwise(0))
        num_meses_distintos_dep_cte_cap = f.sum(f.when(f.col("num_dep_cte_cap") > 0, 1).otherwise(0))
        num_meses_distintos_ret_cte_cap = f.sum(f.when(f.col("num_ret_cte_cap") > 0, 1).otherwise(0))

        tot_mto_dep_cap = f.sum(f.col("mto_dep_cap"))
        tot_num_dep_cap = f.sum(f.col("num_dep_cap"))
        tot_mto_ret_cap = f.sum(f.col("mto_ret_cap"))
        tot_num_ret_cap = f.sum(f.col("num_ret_cap"))
        tot_mto_dep_cte_cap = f.sum(f.col("mto_dep_cte_cap"))
        tot_num_dep_cte_cap = f.sum(f.col("num_dep_cte_cap"))
        tot_mto_ret_cte_cap = f.sum(f.col("mto_ret_cte_cap"))
        tot_num_ret_cte_cap = f.sum(f.col("num_ret_cte_cap"))

        tot_mto_dep_cap_corr = f.sum((f.when(f.col("ord_nom").between(2, 5), f.col("mto_dep_cap")).otherwise(0) / 4) * 6)
        tot_mto_dep_cte_cap_corr = f.sum((f.when(f.col("ord_cte").between(2, 5), f.col("mto_dep_cte_cap")).otherwise(0) / 4) * 6)

        mto_tot_dep_cap_ult_mes = f.sum(f.when(f.col("diff_mes") == 1, f.col("mto_dep_cap")).otherwise(0))
        mto_dep_cte_cap_ult_mes = f.sum(f.when(f.col("diff_mes") == 1, f.col("mto_dep_cte_cap")).otherwise(0))

        TT_inf_aux2_cap = \
        _cap.groupBy(["id_master", "per_ref"]) \
            .agg(ind_activo_cap.alias("ind_activo_cap"),
                ind_activo_afr.alias("ind_activo_afr"),
                ind_activo_cre.alias("ind_activo_cre"),
                ult_dep_cap.alias("ult_dep_cap"),
                ult_ret_cap.alias("ult_ret_cap"),
                ult_dep_cte_cap.alias("ult_dep_cte_cap"),
                ult_ret_cte_cap.alias("ult_ret_cte_cap"),
                num_meses_distintos_movs_cap.alias("num_meses_distintos_movs_cap"),
                num_meses_distintos_dep_cap.alias("num_meses_distintos_dep_cap"),
                num_meses_distintos_ret_cap.alias("num_meses_distintos_ret_cap"),
                num_meses_distintos_movs_cte_cap.alias("num_meses_distintos_movs_cte_cap"),
                num_meses_distintos_dep_cte_cap.alias("num_meses_distintos_dep_cte_cap"),
                num_meses_distintos_ret_cte_cap.alias("num_meses_distintos_ret_cte_cap"),
                tot_mto_dep_cap.alias("tot_mto_dep_cap"),
                tot_num_dep_cap.alias("tot_num_dep_cap"),
                tot_mto_ret_cap.alias("tot_mto_ret_cap"),
                tot_num_ret_cap.alias("tot_num_ret_cap"),
                tot_mto_dep_cte_cap.alias("tot_mto_dep_cte_cap"),
                tot_num_dep_cte_cap.alias("tot_num_dep_cte_cap"),
                tot_mto_ret_cte_cap.alias("tot_mto_ret_cte_cap"),
                tot_num_ret_cte_cap.alias("tot_num_ret_cte_cap"),
                tot_mto_dep_cap_corr.alias("tot_mto_dep_cap_corr"),
                tot_mto_dep_cte_cap_corr.alias("tot_mto_dep_cte_cap_corr"),
                mto_tot_dep_cap_ult_mes.alias("mto_tot_dep_cap_ult_mes"),
                mto_dep_cte_cap_ult_mes.alias("mto_dep_cte_cap_ult_mes")) \
            .orderBy("id_master")
        
        return TT_inf_aux2_cap
    
    
    #paso3: unificacion de informacion de problacion de captacion
    def unif_inf_pob(self,cd_con_cte_ingresos,cd_con_cte_recorrido,TT_inf_aux2_cap):
        TT_inf_feat_cap = \
            TT_inf_aux2_cap.alias("A").join(cd_con_cte_ingresos.alias("B"), \
                (f.col("A.id_master") == f.col("B.id_master")) & \
                    (f.col("A.per_ref") == f.col("B.num_periodo_mes")), how = "left") \
                    .join(cd_con_cte_recorrido.alias("C"), \
                        (f.col("A.id_master") == f.col("C.id_master")) & \
                        (f.col("A.per_ref") == f.col("C.num_periodo_mes")), how = 'left') \
                        .select(f.col("A.id_master"),
                            f.col("A.per_ref"),
                            f.coalesce(f.col("B.cod_flujo"), f.lit('NO DISPONIBLE')).alias("cod_flujo"),
                            f.coalesce(f.col("B.mto_ing_mes"), f.lit(0)).alias("mto_ing_mes"),
                            f.coalesce(f.col("B.num_flujos"), f.lit(0)).alias("num_flujos"),
                            f.coalesce(f.col("B.cod_tipo_nomina") ,f.lit('NO DISPONIBLE')).alias("cod_tipo_nomina"),
                            f.coalesce(f.col("B.cod_tipo_nomina_detalle"), f.lit('NO DISPONIBLE')).alias("cod_tipo_nomina_detalle"),
                            f.coalesce(f.col("B.ind_cre_ov"), f.lit(0)).alias("ind_cre_ov"),
                            f.coalesce(f.col("B.cod_estabilidad_mto"), f.lit('NO DISPONIBLE')).alias("cod_estabilidad_mto"),
                            f.coalesce(f.col("B.cod_estabilidad_periodidad"), f.lit('NO DISPONIBLE')).alias("cod_estabilidad_periodidad"),
                            f.coalesce(f.col("C.cod_perfil_trx"), f.lit('NO DISPONIBLE')).alias("cod_perfil_trx"), \
                            f.coalesce(f.col("C.saldo"), f.lit(0)).alias("saldo"),
                            f.coalesce(f.col("C.potencial"), f.lit(0)).alias("potencial"),
                            f.coalesce(f.col("C.recorrido"), f.lit(0)).alias("recorrido"),
                            f.col("A.ind_activo_cap").alias("ind_activo_cap"),
                            f.col("A.ind_activo_afr").alias("ind_activo_afr"),
                            f.col("A.ind_activo_cre").alias("ind_activo_cre"),
                            f.col("A.ult_dep_cap").alias("ult_dep_cap"),
                            f.col("A.ult_ret_cap").alias("ult_ret_cap"),
                            f.col("A.ult_dep_cte_cap").alias("ult_dep_cte_cap"),
                            f.col("A.ult_ret_cte_cap").alias("ult_ret_cte_cap"),
                            f.col("A.num_meses_distintos_movs_cap").alias("num_meses_distintos_movs_cap"),
                            f.col("A.num_meses_distintos_dep_cap").alias("num_meses_distintos_dep_cap"),
                            f.col("A.num_meses_distintos_ret_cap").alias("num_meses_distintos_ret_cap"),
                            f.col("A.num_meses_distintos_movs_cte_cap").alias("num_meses_distintos_movs_cte_cap"),
                            f.col("A.num_meses_distintos_dep_cte_cap").alias("num_meses_distintos_dep_cte_cap"),
                            f.col("A.num_meses_distintos_ret_cte_cap").alias("num_meses_distintos_ret_cte_cap"),
                            f.col("A.tot_mto_dep_cap").alias("tot_mto_dep_cap"),
                            f.col("A.tot_num_dep_cap").alias("tot_num_dep_cap"),
                            f.col("A.tot_mto_ret_cap").alias("tot_mto_ret_cap"),
                            f.col("A.tot_num_ret_cap").alias("tot_num_ret_cap"),
                            f.col("A.tot_mto_dep_cte_cap").alias("tot_mto_dep_cte_cap"),
                            f.col("A.tot_num_dep_cte_cap").alias("tot_num_dep_cte_cap"),
                            f.col("A.tot_mto_ret_cte_cap").alias("tot_mto_ret_cte_cap"),
                            f.col("A.tot_num_ret_cte_cap").alias("tot_num_ret_cte_cap"),
                            f.col("A.tot_mto_dep_cap_corr").alias("tot_mto_dep_cap_corr"),
                            f.col("A.tot_mto_dep_cte_cap_corr").alias("tot_mto_dep_cte_cap_corr")) \
                        .orderBy(f.col("A.id_master"))
        return TT_inf_feat_cap
              
    def from_csv_to_spark(self,path, file):
        sdf = spark.read.csv(path + file, inferSchema = True, header = True)
        return sdf