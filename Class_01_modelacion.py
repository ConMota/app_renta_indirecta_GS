import os
import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as f
from datetime import datetime




class Train_model():
    
    
    def __init__(self):
        self.str1="First Class"
    
    
    def export_model(self,VAR_TARGET):
        
        path_train_data = 'hdfs://srvhdfsha/user/297397/projects_dev/cognodata/renta_indirecta/app/train/data/TT_train_feat_ren_ind/'
        csv_train = 'part-00000-c7670210-7440-4f28-b0dc-46d7fd7ce1a4-c000.csv' 
        TRAIN_FEAT_REN_IND = spark.read.format("csv").option("header", True).option("inferSchema",True).load(path_train_data+csv_train)
        
        vector_features = self.transform_data(VAR_TARGET,TRAIN_FEAT_REN_IND)
        model_RM = self.model_RandomForest(VAR_TARGET,vector_features)

        #factor_ajuste = self.factor_ajuste(VAR_TARGET,TRAIN_FEAT_REN_IND,model_RM['path_model'])
        if model_RM['status'] == 'success':
            return {'status':'success','path_model': model_RM['path_model'],'metrics':model_RM['metrics']}
            
        
        #if model_RM['status'] == 'success' and factor_ajuste['status'] == 'success':
        #    return {'status':'success','path_model': model_RM['path_model'],'metrics':model_RM['metrics']}
        #elif model_RM['status'] == 'error' or factor_ajuste['status'] == 'error':
        #    return {'status':'error','msg':model_RM['msg']}
        else:
            return {'status':'error','msg':'Contact support'}
    
    def transform_data(self,VAR_TARGET,TRAIN_FEAT_REN_IND):
        
        data_ren = TRAIN_FEAT_REN_IND[[VAR_TARGET, 'saldo', 'potencial', 'recorrido', 'edad']]
        data_ren = data_ren.withColumn('log_'+VAR_TARGET,f.log10(f.col(VAR_TARGET)+1))
        data_ren = data_ren.drop(VAR_TARGET)
        data_ren = data_ren.dropDuplicates()
        data_ren = data_ren.filter((f.col('edad')>= 18) & (f.col('edad') < 100))
        vector_features = VectorAssembler(inputCols=['saldo', 'potencial', 'recorrido', 'edad'], outputCol='features').transform(data_ren)
        del TRAIN_FEAT_REN_IND
        return vector_features
    
    def model_RandomForest(self,VAR_TARGET,vector_features):
        
        #try:
        path_dev = 'hdfs://srvhdfsha/user/297397/projects_dev/cognodata/renta_indirecta/app/infierence/models'
        
        training, test= vector_features.randomSplit([0.8,0.2])

        # Train a RandomForest model.
        rf = RandomForestRegressor(
                featuresCol='features',
                labelCol='log_'+VAR_TARGET
                )
        

        model=rf.fit(training)
        cols=['saldo','potencial','recorrido','edad']
        labels_importance={cols[i]: list(model.featureImportances)[i] for i in range(len(cols))}
        
        predictions = model.transform(test)
        predictions.repartition(1).write.format('com.databricks.spark.csv')\
                .option('header',True) \
                .mode('overwrite') \
                .save(path_dev)
        
        #valuesAndPreds = predictions.select(['log_'+VAR_TARGET, 'prediction'])
        #valuesAndPreds = valuesAndPreds.rdd.map(tuple)
        #metrics = RegressionMetrics(valuesAndPreds)

        #metrics_dict = {
        #    'MAE': metrics.meanAbsoluteError,
        #    'MSE': metrics.meanSquaredError,
        #    'R2': metrics.r2,
        #    'RMSE': metrics.rootMeanSquaredError,
        #    'expleinedVariance':metrics.explainedVariance
        #}
        #print(metrics_dict)
        
        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(labelCol='log_'+VAR_TARGET, predictionCol="prediction", metricName="rmse")
        metrics = evaluator.evaluate(predictions)
        

        
        #today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        save_model =   path_dev #+ '/{}'.format(today)
        #os.mkdir(save_model)
        
        

        #file_info = open(save_model + "/file_info.txt", "w")
        #file_info.write("===== Información modelo renta indirecta =====")
        #file_info.write("---------------------------------")
        #file_info.write("Descripción Modelo")
        #file_info.write(model)
        #file_info.write("---------------------------------")
        #file_info.write("Atributos de importancia")
        #file_info.write(labels_importance)
        #file_info.write("---------------------------------")
        #file_info.write('Metrics')
        #file_info.write("RMSE on test = %g" % metrics)
        #file_info.write('======= Sumary ================')
        #file_info.close()


        rf.write().overwrite().save(save_model)

        
        respond = {'status':'success','path_model':save_model,'metrics':metrics}
        return respond
        #except:
        #    return {'status':'error','msg':'error in model RandomForest'}


    def factor_ajuste(self,VAR_TARGET,TRAIN_FEAT_REN_IND,save_model):   
            
            try:
                df_succ = spark.read.csv('hdfs://srvhdfsha/user/297397/Models_Renta_Indirecta/data/success.csv',header=True)
                
                data_succ = df_succ[ df_succ.Nivel == 'Código postal' ]
                data_succ = data_succ[['Código Postal', 'Familias', 'Ingreso familiar promedio $', 'Población 2019', 'Población Económicamente Inactiva (PEI)']]
                
                data_succ['ingreso_prom_succ'] = data_succ['Ingreso familiar promedio $'] / ((data_succ['Población 2019'] - data_succ['Población Económicamente Inactiva (PEI)'])/ data_succ['Familias'])
                data_succ = data_succ.rename(columns = {'Código Postal':'cod_postal'})
               
                
                #data_succ.fillna(0, inplace=True)
                #data_succ['Población Económicamente Inactiva (PEI)'] = np.fmax(data_succ['Población Económicamente Inactiva (PEI)'], 0)
                #data_succ = data_succ[data_succ['Población 2019'] - data_succ['Población Económicamente Inactiva (PEI)'] > 0]
                
                df_ren = df_ren.merge( data_succ[['cod_postal', 'ingreso_prom_succ']], how='left', on='cod_postal' )
                df_ren['ingreso_prom_succ'] = np.where( df_ren['ingreso_prom_succ'].isna(), 0, df_ren['ingreso_prom_succ'] )
                
                df_comp = df_ren[[VAR_TARGET, 'ingreso_prom_succ', 'cod_postal']]
                df_comp = df_comp[ (df_comp.cod_postal != 99999) & (df_comp.ingreso_prom_succ > 0) ]
                factor = df_comp['mto_ing_mes'].median() / df_comp['ingreso_prom_succ'].median()
                factor = round(factor, 4)
                dict_fat = {'factor' : df_comp['mto_ing_mes'].median() / df_comp['ingreso_prom_succ'].median()}
                save_factor_ajuste = save_model + '/factor_ajuste.pkl'

                joblib.dump(value=dict_fat, filename= save_factor_ajuste)
                respond = {'status':'success','path_model':save_factor_ajuste}
                return respond
            except:    
                respond = {'status':'error','msg':'Error in factor ajuste'}
                return respond
            
            






