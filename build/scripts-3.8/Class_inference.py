import os
import joblib
import pandas as pd
import numpy as np
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



class Inference_RI():
    def __init__(self):
        self.str1="First Class"

    def output_inference(self,INF_FEAT_REN_IND):
        # Carga de todos los insumos
        
        # Informacion de los clientes
        df_ren = INF_FEAT_REN_IND.toPandas()
        # success tabla INEGI
        df_succ = pd.read_csv('Inputs/success.csv')
        # Modelo entrenado

        youngest_model = max(os.listdir('app/inference/models/'))    
        path_model = 'app/inference/models/{}/model.pkl'.format(youngest_model)
        regr = joblib.load(filename=path_model)
        # Factor de ajuste
        path_fact = 'app/inference/models/{}/factor_ajuste.pkl'.format(youngest_model)
        fact_INEGI = joblib.load(filename=path_fact)
        
        '''
        Para obtener el promedio de ingresos reportados por el INEGI, se toman solo las variables 
        relevantes para el análisis, y se realiza el cálculo del ingreso promedio por código postal 
        dejando un registro por código postal
        '''
        
        data_succ = df_succ[ df_succ.Nivel == 'Código postal']
        data_succ = data_succ[['Código Postal', 'Familias', 'Ingreso familiar promedio $', 'Población 2019', 'Población Económicamente Inactiva (PEI)']]
        data_succ['ingreso_prom_succ'] = data_succ['Ingreso familiar promedio $'] / ((data_succ['Población 2019'] - data_succ['Población Económicamente Inactiva (PEI)'])/ data_succ['Familias'])    
        data_succ = data_succ.rename(columns = {'Código Postal':'cod_postal'})
        

        
        df_ren = df_ren.merge( data_succ[['cod_postal', 'ingreso_prom_succ']], how='left', on='cod_postal' )
        df_ren['ingreso_prom_succ'] = np.where( df_ren['ingreso_prom_succ'].isna(), 0, df_ren['ingreso_prom_succ'] )
        #print(df_ren[['id_master','cod_postal','ingreso_prom_succ']].head())
        
        # inferencia del ingreso mensual
        df_RFM = df_ren.copy()
        # se aplica la estimacion de potenciales solo a los clientes de captacion
        df_cap = df_RFM[ df_RFM['ind_activo_cap'] == 1 ].copy()
        df_otr = df_RFM[ df_RFM['ind_activo_cap'] == 0 ].copy()
        df_otr['regr'] = 0
        
        y_inferencia = regr.predict( df_cap[['potencial', 'recorrido', 'saldo', 'edad']] )
        df_inferencia = pd.DataFrame(y_inferencia, columns=['regr'])
        df_inferencia['regr'] = 10 ** df_inferencia['regr'] - 1
        df_cap = pd.concat( [df_cap.reset_index(drop=True), df_inferencia ], axis=1 ).reset_index(drop=True)

        # se unen los DF de captacion y otros
        df_RFM = pd.concat( [df_cap.reset_index(drop=True), df_otr.reset_index(drop=True) ], axis=0 ).reset_index(drop=True)
        df_RFM['ingreso_prom_succ'] = df_RFM['ingreso_prom_succ'] * fact_INEGI['factor']
        df_RFM['ing_sdm'] = df_RFM[['regr', 'ingreso_prom_succ']].max(axis=1)
        
        ## score de recencia
        max_rec = df_RFM[ df_RFM.ult_dep_cap < 99 ]['ult_dep_cap'].max()

        df_RFM['score_rec_dep_cap'] = np.where( df_RFM['ult_dep_cap'] == 99, 0, ( max_rec - ( df_RFM['ult_dep_cap'] - 1 ) ) / max_rec )
        df_RFM['score_rec_ret_cap'] = np.where( df_RFM['ult_ret_cap'] == 99, 0, ( max_rec - ( df_RFM['ult_ret_cap'] - 1 ) ) / max_rec )
        df_RFM['score_rec_dep_cte_cap'] = np.where( df_RFM['ult_dep_cte_cap'] == 99, 0, ( max_rec - ( df_RFM['ult_dep_cte_cap'] - 1 ) ) / max_rec )
        df_RFM['score_rec_ret_cte_cap'] = np.where( df_RFM['ult_ret_cte_cap'] == 99, 0, ( max_rec - ( df_RFM['ult_ret_cte_cap'] - 1 ) ) / max_rec )
        df_RFM['score_rec_cpa_div'] = np.where( df_RFM['ult_cpa_div'] == 99, 0, ( max_rec - ( df_RFM['ult_cpa_div'] - 1 ) ) / max_rec )
        df_RFM['score_rec_vta_div'] = np.where( df_RFM['ult_vta_div'] == 99, 0, ( max_rec - ( df_RFM['ult_vta_div'] - 1 ) ) / max_rec )
        df_RFM['score_rec_pag_dex'] = np.where( df_RFM['ult_pag_dex'] == 99, 0, ( max_rec - ( df_RFM['ult_pag_dex'] - 1 ) ) / max_rec )
        df_RFM['score_rec_env_dex'] = np.where( df_RFM['ult_env_dex'] == 99, 0, ( max_rec - ( df_RFM['ult_env_dex'] - 1 ) ) / max_rec )
        df_RFM['score_rec_pag_rem'] = np.where( df_RFM['ult_pag_rem'] == 99, 0, ( max_rec - ( df_RFM['ult_pag_rem'] - 1 ) ) / max_rec )
        df_RFM['score_rec_env_rem'] = np.where( df_RFM['ult_env_rem'] == 99, 0, ( max_rec - ( df_RFM['ult_env_rem'] - 1 ) ) / max_rec )

        pd.options.display.max_columns = None
        df_RFM.filter(regex= 'id_master|ult_|score_rec')

        ## score frecuencia
        max_frec = df_RFM[ df_RFM.num_meses_distintos_dep_cap < 99 ]['num_meses_distintos_dep_cap'].max()

        df_RFM['score_frec_movs_cap'] = np.where( df_RFM['num_meses_distintos_movs_cap'] == 0, 0, df_RFM['num_meses_distintos_movs_cap'] / max_frec )
        df_RFM['score_frec_dep_cap'] = np.where( df_RFM['num_meses_distintos_dep_cap'] == 0, 0, df_RFM['num_meses_distintos_dep_cap'] / max_frec )
        df_RFM['score_frec_ret_cap'] = np.where( df_RFM['num_meses_distintos_ret_cap'] == 0, 0, df_RFM['num_meses_distintos_ret_cap'] / max_frec )
        df_RFM['score_frec_movs_cte_cap'] = np.where( df_RFM['num_meses_distintos_movs_cte_cap'] == 0, 0, df_RFM['num_meses_distintos_movs_cte_cap'] / max_frec )
        df_RFM['score_frec_dep_cte_cap'] = np.where( df_RFM['num_meses_distintos_dep_cte_cap'] == 0, 0, df_RFM['num_meses_distintos_dep_cte_cap'] / max_frec )
        df_RFM['score_frec_ret_cte_cap'] = np.where( df_RFM['num_meses_distintos_ret_cte_cap'] == 0, 0, df_RFM['num_meses_distintos_ret_cte_cap'] / max_frec )
        df_RFM['score_frec_movs_div'] = np.where( df_RFM['num_meses_distintos_movs_div'] == 0, 0, df_RFM['num_meses_distintos_movs_div'] / max_frec )
        df_RFM['score_frec_cpa_div'] = np.where( df_RFM['num_meses_distintos_cpa_div'] == 0, 0, df_RFM['num_meses_distintos_cpa_div'] / max_frec )
        df_RFM['score_frec_vta_div'] = np.where( df_RFM['num_meses_distintos_vta_div'] == 0, 0, df_RFM['num_meses_distintos_vta_div'] / max_frec )
        df_RFM['score_frec_movs_rem'] = np.where( df_RFM['num_meses_distintos_movs_rem'] == 0, 0, df_RFM['num_meses_distintos_movs_rem'] / max_frec )
        df_RFM['score_frec_pag_rem'] = np.where( df_RFM['num_meses_distintos_pag_rem'] == 0, 0, df_RFM['num_meses_distintos_pag_rem'] / max_frec )
        df_RFM['score_frec_env_rem'] = np.where( df_RFM['num_meses_distintos_env_rem'] == 0, 0, df_RFM['num_meses_distintos_env_rem'] / max_frec )
        df_RFM['score_frec_movs_dex'] = np.where( df_RFM['num_meses_distintos_movs_dex'] == 0, 0, df_RFM['num_meses_distintos_movs_dex'] / max_frec )
        df_RFM['score_frec_pag_dex'] = np.where( df_RFM['num_meses_distintos_pag_dex'] == 0, 0, df_RFM['num_meses_distintos_pag_dex'] / max_frec )
        df_RFM['score_frec_env_dex'] = np.where( df_RFM['num_meses_distintos_env_dex'] == 0, 0, df_RFM['num_meses_distintos_env_dex'] / max_frec )

        pd.options.display.max_columns = None
        df_RFM.filter(regex= 'id_master|num_meses_distintos_|score_frec')

        ## RFM Captacion
        threshhold_score_cap = 0.5
        df_RFM['score_dep_cap'] = np.where( (df_RFM['score_frec_dep_cte_cap'] <= threshhold_score_cap) | (df_RFM['score_rec_dep_cte_cap'] <= threshhold_score_cap) | (df_RFM['score_rec_dep_cte_cap'] < df_RFM ['score_frec_dep_cte_cap']), 0, (df_RFM['score_rec_dep_cte_cap'] + df_RFM['score_frec_dep_cte_cap']) / 2  )
        df_RFM['RFM_cap'] = np.where( df_RFM['score_dep_cap'] == 0, 0, ( df_RFM['tot_mto_dep_cte_cap_corr'] / df_RFM['num_meses_distintos_dep_cap'] ) )
        df_RFM[ df_RFM.RFM_cap >0 ].filter(regex='id_master|score_frec_dep_cap|score_rec_dep_cap|score_dep_cap|tot_mto_dep_cap|RFM_cap') 

        ## RFM Divisas
        threshhold_score_div = 0.5
        df_RFM['score_cpa_div'] = np.where( (df_RFM['score_frec_cpa_div'] <= threshhold_score_div) | (df_RFM['score_rec_cpa_div'] <= threshhold_score_div) | (df_RFM['score_rec_cpa_div'] < df_RFM['score_frec_cpa_div']), 0, 
            (df_RFM['score_rec_cpa_div'] + df_RFM['score_frec_cpa_div']) / 2  )

        df_RFM['score_vta_div'] = np.where( (df_RFM['score_frec_vta_div'] <= threshhold_score_div) | (df_RFM['score_rec_vta_div'] <= threshhold_score_div) | (df_RFM['score_rec_vta_div'] < df_RFM['score_frec_vta_div']), 0, 
            (df_RFM['score_rec_vta_div'] + df_RFM['score_frec_vta_div']) / 2  )


        df_RFM['RFM_div'] = np.where( df_RFM['score_cpa_div'] > threshhold_score_div, 
                df_RFM['score_cpa_div']*( ( df_RFM['tot_mto_cpa_div'] -  df_RFM['tot_mto_vta_div'] ) / df_RFM['num_meses_distintos_cpa_div'] ), 0 )
                                    

        df_RFM.filter(regex='id_master|mto_ing_mes|score_frec_movs_div|score_rec_cpa_div|score_rec_vta_div|score_movs_div|score_cpa_div|score_vta_div|tot_mto_vta_div|tot_mto_cpa_div|num_meses_distintos_cpa_div|num_meses_distintos_vta_div|RFM_div')[ df_RFM.RFM_div > 0]
        
        
        ## RFM DEX
        threshhold_score_dex = 0.5

        df_RFM['score_pag_dex'] = np.where( (df_RFM['score_frec_pag_dex'] <= threshhold_score_dex) | (df_RFM['score_rec_pag_dex'] <= threshhold_score_dex) | (df_RFM['score_rec_pag_dex'] < df_RFM['score_frec_pag_dex']), 0, 
            (df_RFM['score_rec_pag_dex'] + df_RFM['score_frec_pag_dex']) / 2  )

        df_RFM['score_env_dex'] = np.where( (df_RFM['score_frec_env_dex'] <= threshhold_score_dex) | (df_RFM['score_rec_env_dex'] <= threshhold_score_dex) | (df_RFM['score_rec_env_dex'] < df_RFM['score_frec_env_dex']), 0, 
            (df_RFM['score_rec_env_dex'] + df_RFM['score_frec_env_dex']) / 2  )

        df_RFM['RFM_dex'] = np.where( df_RFM['score_pag_dex'] > threshhold_score_dex, 
                                    df_RFM['score_pag_dex']*( ( df_RFM['tot_mto_pag_dex'] -  df_RFM['tot_mto_env_dex'] ) / df_RFM['num_meses_distintos_pag_dex'] ), 0) 
                                    
        df_RFM.filter(regex='id_master|mto_ing_mes|score_frec_movs_dex|score_rec_pag_dex|score_rec_env_dex|score_movs_dex|score_pag_dex|score_env_dex|tot_mto_env_dex|tot_mto_pag_dex|num_meses_distintos_pag_dex|num_meses_distintos_env_dex|RFM_dex')[ df_RFM.RFM_dex > 0]


        ## RFM remesas
        threshhold_score_rem = 0.5

        df_RFM['score_pag_rem'] = np.where( (df_RFM['score_frec_pag_rem'] <= threshhold_score_rem) | (df_RFM['score_rec_pag_rem'] <= threshhold_score_rem) | (df_RFM['score_rec_pag_rem'] < df_RFM['score_frec_pag_rem']), 0, 
            (df_RFM['score_rec_pag_rem'] + df_RFM['score_frec_pag_rem']) / 2  )

        df_RFM['score_env_rem'] = np.where( (df_RFM['score_frec_env_rem'] <= threshhold_score_rem) | (df_RFM['score_rec_env_rem'] <= threshhold_score_rem) | (df_RFM['score_rec_env_rem'] < df_RFM['score_frec_env_rem']), 0, 
            (df_RFM['score_rec_env_rem'] + df_RFM['score_frec_env_rem']) / 2  )

        df_RFM['RFM_rem'] = np.where( df_RFM['score_pag_rem'] > threshhold_score_rem, 
                                    df_RFM['score_pag_rem']*( ( df_RFM['tot_mto_pag_rem'] -  df_RFM['tot_mto_env_rem'] ) / df_RFM['num_meses_distintos_pag_rem'] ), 0)
                                    

        df_RFM.filter(regex='id_master|mto_ing_mes|score_frec_movs_rem|score_rec_pag_rem|score_rec_env_rem|score_movs_rem|score_pag_rem|score_env_rem|tot_mto_env_rem|tot_mto_pag_rem|num_meses_distintos_pag_rem|num_meses_distintos_env_rem|RFM_rem')[ df_RFM.RFM_rem > 0 ]
        df_RFM['RFM_servicios'] = df_RFM[['RFM_div', 'RFM_dex', 'RFM_rem']].sum(axis=1)
        df_RFM[['RFM_div', 'RFM_dex', 'RFM_rem', 'RFM_servicios']][ (df_RFM['RFM_dex'] > 0) &  (df_RFM['RFM_rem'] > 0) ]

        ### se calculan los ingresos extraordinarios de captacion
        df_RFM['ing_extraordinarios_cap'] = np.where(df_RFM['RFM_cap'] - df_RFM['mto_ing_mes'] > 0, df_RFM['RFM_cap'] - df_RFM['mto_ing_mes'], 0) * df_RFM['score_dep_cap']
        df_RFM['ingresos_otros'] = df_RFM[ ['mto_ing_mes', 'RFM_cap', 'RFM_servicios', 'regr' ] ].max(axis=1)
        df_RFM['ingresos_otros'] = np.where( df_RFM['ingresos_otros'] > 100, df_RFM['ingresos_otros'], df_RFM['ingreso_prom_succ'] )
        df_RFM['etiqueta_otros'] = np.where( df_RFM['ingresos_otros'] == df_RFM['mto_ing_mes'], 'RENTA DIRECTA', 
                                    np.where( df_RFM['ingresos_otros'] == df_RFM['RFM_cap'], 'RFM(CAPTACION)', 
                                    np.where( df_RFM['ingresos_otros'] == df_RFM['RFM_servicios'], 'SERVICIOS', 
                                    np.where( df_RFM['ingresos_otros'] == df_RFM['ingreso_prom_succ'], 'INEGI', 
                                    'POTENCIALES'
                                    ) 
                                ) 
                            ) 
                        )
     




        df_RFM['valor_renta'] = np.nan
        ## Nomina observada;
        df_RFM['valor_renta'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '01 OBSERVADA'), 
            df_RFM['mto_ing_mes'] + np.where(df_RFM['RFM_cap'] >= df_RFM['RFM_servicios'], df_RFM['RFM_cap'], df_RFM['RFM_servicios']), df_RFM['valor_renta'] ) 

        ## nomina estimada; efectivo;
        df_RFM['valor_renta'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '02 ESTIMADA') & (df_RFM['cod_flujo'] == 'EFECTIVO'),
            np.where( df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'], df_RFM['RFM_servicios']), 
            df_RFM['valor_renta'])
        ## nomina estimada; otros flujos;
        df_RFM['valor_renta'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '02 ESTIMADA') & (df_RFM['cod_flujo'] != 'EFECTIVO'), df_RFM['mto_ing_mes'] + 
        np.where( df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], df_RFM['ing_extraordinarios_cap'], df_RFM['RFM_servicios']), df_RFM['valor_renta']) 

        ## nomina estimada; al menos un flujo
        print (df_RFM['num_flujos'])
        df_RFM['num_flujos'] = pd.notnull(df_RFM.num_flujos)
        print (df_RFM['num_flujos'])
        
        df_RFM['valor_renta'] = np.where( (df_RFM['cod_tipo_nomina'] == '03 NO ESTIMADA') & (df_RFM['num_flujos'].astype(int) >= 1),
            np.where( df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'], df_RFM['RFM_servicios']), 
            df_RFM['valor_renta'])

        ## Sin flujo relevante o sin estimacion de renta directa
        df_RFM['valor_renta'] = np.where( (df_RFM['cod_tipo_nomina'] == 'VACIO') | (df_RFM['num_flujos'] == 0), df_RFM['ingresos_otros'], df_RFM['valor_renta']) 


        df_RFM['regla'] = np.nan
        ## Nomina observada; 
        df_RFM['regla'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '01 OBSERVADA'), np.where(df_RFM['RFM_cap'] >= df_RFM['RFM_servicios'], np.where(df_RFM['RFM_cap'] > 0, 'RENTA DIRECTA + RFM(CAPTACION)', 'RENTA DIRECTA'), np.where(df_RFM['RFM_servicios'] > 0, 'RENTA DIRECTA + SERVICIOS', 'RENTA DIRECTA')), df_RFM['regla'] ) 

        ## nomina estimada; efectivo;
        df_RFM['regla'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '02 ESTIMADA') & (df_RFM['cod_flujo'] == 'EFECTIVO'),
            np.where( df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], np.where(df_RFM['ing_extraordinarios_cap'] > 0, 'RENTA DIRECTA + INGRESOS EXTRAORDINARIOS CAPTACION', 'RENTA DIRECTA'), np.where(df_RFM['RFM_servicios'] > 0, 'SERVICIOS', '') ), df_RFM['regla'])
        ## nomina estimada; otros flujos;
        df_RFM['regla'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '02 ESTIMADA') & (df_RFM['cod_flujo'] != 'EFECTIVO'), np.where( df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], np.where(df_RFM['ing_extraordinarios_cap'] > 0, 'RENTA DIRECTA + INGRESOS EXTRAORDINARIOS CAPTACION', 'RENTA DIRECTA') , np.where(df_RFM['RFM_servicios'] > 0, 'RENTA DIRECTA + SERVICIOS', 'RENTA DIRECTA')), df_RFM['regla']) 

        ## no estimada; al menos un flujo
        df_RFM['regla'] = np.where( (df_RFM['cod_tipo_nomina'] == '03 NO ESTIMADA') & (df_RFM['num_flujos'] >= 1),
            np.where( df_RFM['mto_ing_mes'] + df_RFM['ing_extraordinarios_cap'] >= df_RFM['RFM_servicios'], np.where(df_RFM['ing_extraordinarios_cap'] > 0, 'RENTA DIRECTA + INGRESOS EXTRAORDINARIOS CAPTACION', 'RENTA DIRECTA'), np.where(df_RFM['RFM_servicios'] > 0, 'SERVICIOS', '')), df_RFM['regla'])

        ## Sin flujo relevante o sin estimacion de renta directa
        df_RFM['regla'] = np.where( (df_RFM['cod_tipo_nomina'] == 'VACIO') | (df_RFM['num_flujos'] == 0), np.where(df_RFM['etiqueta_otros']=='RENTA DIRECTA', 'CAPACIDAD DE PAGO', df_RFM['etiqueta_otros']), df_RFM['regla'])


        df_RFM['etiqueta'] = np.nan
        ## Nomina observada; 
        df_RFM['etiqueta'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '01 OBSERVADA'), '01 NOMINA OBSERVADA', df_RFM['etiqueta'] ) 

        ## nomina estimada
        df_RFM['etiqueta'] = np.where( (df_RFM['cod_tipo_nomina'] ==  '02 ESTIMADA'), '02 NOMINA ESTIMADA', df_RFM['etiqueta']) 

        ## no estimada; al menos un flujo
        df_RFM['etiqueta'] = np.where( (df_RFM['cod_tipo_nomina'] == '03 NO ESTIMADA') & (df_RFM['num_flujos'] >= 1), '03 NOMINA NO ESTIMADA', df_RFM['etiqueta'])

        ## Sin flujo relevante o sin estimacion de renta directa
        df_RFM['etiqueta'] = np.where( (df_RFM['cod_tipo_nomina'] == 'VACIO') | (df_RFM['num_flujos'] == 0), '04 SIN RENTA DIRECTA', df_RFM['etiqueta']) 



        ## variables indicadoras
        df_RFM['RENTA_DIRECTA'] = np.where( ( ( df_RFM['regla'].str.contains('RENTA DIRECTA') ) | ( df_RFM['regla'].str.contains('CAPACIDAD DE PAGO') ) ) & (df_RFM['mto_ing_mes'] > 0), 1, 0 ) 
        df_RFM['INGRESOS_EXTRAORDINARIOS_CAP'] = np.where( (df_RFM['regla'].str.contains('INGRESOS EXTRAORDINARIOS CAPTACION')) & (df_RFM['ing_extraordinarios_cap'] > 0), 1, 0 )
        df_RFM['RFM_CAP'] = np.where( df_RFM['regla'].str.contains('RFM(CAPTACION)') & (df_RFM['RFM_cap'] > 0), 1, 0 ) 
        df_RFM['SERVICIOS'] = np.where( df_RFM['regla'].str.contains('SERVICIOS') & (df_RFM['RFM_servicios'] > 0), 1, 0 )
        df_RFM['POTENCIALES'] = np.where( df_RFM['regla'].str.contains('POTENCIALES') & (df_RFM['regr'] > 0), 1, 0 )
        df_RFM['INEGI'] = np.where( df_RFM['regla'].str.contains('INEGI') & (df_RFM['ingreso_prom_succ'] > 0), 1, 0 ) 
        df_RFM['DIVISAS'] = np.where( (df_RFM['SERVICIOS'] == 1) & (df_RFM['RFM_div'] > 0), 1, 0 )
        df_RFM['DEX'] = np.where( (df_RFM['SERVICIOS'] == 1) & (df_RFM['RFM_dex'] > 0), 1, 0 )
        df_RFM['REMESAS'] = np.where( (df_RFM['SERVICIOS'] == 1) & (df_RFM['RFM_rem'] > 0), 1, 0 )
        df_RFM['mto_div'] = np.where( df_RFM['tot_mto_cpa_div'] -  df_RFM['tot_mto_vta_div'] > 0, df_RFM['tot_mto_cpa_div'] -  df_RFM['tot_mto_vta_div'], 0 )
        df_RFM['mto_dex'] = np.where( df_RFM['tot_mto_pag_dex'] -  df_RFM['tot_mto_env_dex'] > 0, df_RFM['tot_mto_pag_dex'] -  df_RFM['tot_mto_env_dex'], 0 )
        df_RFM['mto_rem'] = np.where( df_RFM['tot_mto_pag_rem'] -  df_RFM['tot_mto_env_rem'] > 0, df_RFM['tot_mto_pag_rem'] -  df_RFM['tot_mto_env_rem'], 0 )

        df_RFM = df_RFM.rename(columns={
            'valor_renta' : 'ing_renta_indirecta',
            'etiqueta' : 'cod_grupo_renta',
            'regla' : 'cod_ingreso_renta',
            'mto_ing_mes' : 'ing_renta_directa',
            'RFM_cap' : 'ing_rfm_cap_tot',
            'RFM_servicios' : 'ing_rfm_servicios',
            'RFM_div' : 'ing_rfm_div',  
            'RFM_dex' : 'ing_rfm_dex',  
            'RFM_rem' : 'ing_rfm_rem',
            'ing_extraordinarios_cap' : 'ing_rfm_cap_dif',
            'regr' : 'ing_pot',
            'ingreso_prom_succ' : 'ing_inegi',
            'RENTA_DIRECTA' : 'ind_renta_directa',
            'INGRESOS_EXTRAORDINARIOS_CAP' : 'ind_rfm_cap_dif',
            'RFM_CAP' : 'ind_rfm_cap_tot',
            'SERVICIOS' : 'ind_rfm_servicios',
            'POTENCIALES' : 'ind_pot',
            'INEGI' : 'ind_inegi',
            'DIVISAS' : 'ind_rfm_div',
            'DEX' : 'ind_rfm_dex',
            'REMESAS' : 'ind_rfm_rem',
            'ult_dep_cap' : 'rec_cap',
            'ult_cpa_div' : 'rec_div',
            'ult_pag_rem' : 'rec_rem',
            'ult_pag_dex' : 'rec_dex',
            'num_meses_distintos_dep_cte_cap' : 'frec_cap',
            'num_meses_distintos_cpa_div' : 'frec_div',
            'num_meses_distintos_pag_rem' : 'frec_rem',
            'num_meses_distintos_pag_dex' : 'frec_dex',
            'tot_mto_dep_cap_corr' : 'mto_cap',
            'tot_mto_cpa_div' : 'mto_cpa_div',
            'tot_mto_vta_div' : 'mto_vta_div',
            'tot_mto_pag_dex' : 'mto_pag_dex',
            'tot_mto_env_dex' : 'mto_env_dex',
            'tot_mto_pag_rem' : 'mto_pag_rem',
            'tot_mto_env_rem' : 'mto_env_rem',
            'score_dep_cap' : 'score_cap',
            'score_cpa_div' : 'score_div',
            'score_pag_dex' : 'score_dex',
            'score_pag_rem' : 'score_rem',
            'per_ref' : 'num_periodo_mes'
        })


        # Se formatea a 4 decimales las columnas de montos
        lista_mtos = ['ing_renta_indirecta', 'ing_renta_directa', 'ing_rfm_cap_tot', 'ing_rfm_cap_dif', 'ing_rfm_servicios', 'ing_rfm_rem', 'ing_rfm_div', 'ing_rfm_dex', 'ing_pot', 'ing_inegi', 'mto_cap', 'score_cap', 'mto_rem', 'score_rem', 'mto_div', 'score_div', 'mto_dex', 'score_dex']
        for l in lista_mtos:
            df_RFM[l] = round( df_RFM[l], 4 )
            
            
        df_RFM['ing_renta_indirecta']
        
        # se ordenan las variables para el archivo de salida
        df_RFM = df_RFM[[ 'id_master', 'cod_grupo_renta', 'cod_ingreso_renta', 'cod_tipo_nomina', 'cod_flujo', 'ind_activo_cap', 'ind_activo_dex', 'ind_activo_rem', 'ind_activo_div', 'ind_activo_afr', 'ind_activo_cre', 'ing_renta_indirecta', 'ing_renta_directa', 'ing_rfm_cap_tot', 'ing_rfm_cap_dif', 'ing_rfm_servicios', 'ing_rfm_rem', 'ing_rfm_div', 'ing_rfm_dex', 'ing_pot', 'ing_inegi', 'ind_renta_directa', 'ind_rfm_cap_tot', 'ind_rfm_cap_dif', 'ind_rfm_servicios', 'ind_rfm_rem', 'ind_rfm_div', 'ind_rfm_dex', 'ind_pot', 'ind_inegi', 'rec_cap', 'frec_cap', 'mto_cap', 'score_cap', 'rec_rem', 'frec_rem', 'mto_rem', 'score_rem', 'rec_div', 'frec_div', 'mto_div', 'score_div', 'rec_dex', 'frec_dex', 'mto_dex', 'score_dex', 'edad', 'cod_postal', 'num_periodo_mes' ]]

        ruta_salida = 'app/inference/output/salida.csv'
        df_RFM.to_csv(ruta_salida, index=False)
        df_RFM[['ing_renta_indirecta']].describe()
        
        df_conteos = df_RFM.groupby([ 'cod_grupo_renta', 'cod_ingreso_renta' ])[['ing_renta_indirecta']].describe(percentiles=[0.01, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])['ing_renta_indirecta'].reset_index()
        df_conteos['prop'] = 100* df_conteos['count'] / df_conteos['count'].sum()
        df_conteos = df_conteos[[ 'cod_grupo_renta', 'cod_ingreso_renta', 'count', 'prop', 'mean', 'std', 'min', '1%', '5%', '10%', '15%', '25%', '35%', '50%', '60%', '75%', '90%', '95%', '99%', 'max' ]]
        
        
        ruta_reporte = 'app/inference/output/reporte.csv'
        df_conteos.to_csv(ruta_reporte, index=False)