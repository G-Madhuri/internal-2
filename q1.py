import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

num_samples=300
X,y=make_classification(n_samples=num_samples,n_features=6,n_classes=2,random_state=42)
column_names=['fbs','age','target','thalach','chol','restecg']
heartdisease=pd.DataFrame(X,columns=column_names)
heartdisease['target']=y
heartdisease['age']=np.round(((heartdisease['age']-heartdisease['age'].min())/(heartdisease['age'].max()-heartdisease['age'].min()))*50 +30).astype(int)
heartdisease[column_names]=heartdisease[column_names].apply(lambda x:np.round(x).astype(int))
model=DiscreteBayesianNetwork([('age','fbs'),('fbs','target'),('fbs','target'),('target','chol'),('target','thalach'),('target','restecg')])
model.fit(heartdisease,estimator=MaximumLikelihoodEstimator)
heartdisease_infer=VariableElimination(model)
res=heartdisease_infer.query(variables=['target'],evidence={'age':30})
print("Bayesian network probabilities are: ")
print(res)