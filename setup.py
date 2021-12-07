import setuptools

setuptools.setup(
     name='AppRentaIndirectaGS',  #nombre del paquete
     version='1.0', #versión
     scripts=['Class_01_obs.py','Class_02_feat.py','Class_01_obs_cap.py','Class_02_obs_env.py','Class_03_obs_div.py','Class_04_feat_cap.py','Class_05_feat_env.py','Class_06_feat_div.py','Class_07_unification.py','Class_inference.py','Class_01_modelacion.py'] , #nombre del ejecutable
     author="Team IA", #autor
     author_email="mizantha.mota@elektra.com.mx", #email
     description="App renta indirecta", #Breve descripción
     packages=setuptools.find_packages(), #buscamos todas las dependecias necesarias para que tu paquete funcione (por ejemplo numpy, scipy, etc.)
) 

