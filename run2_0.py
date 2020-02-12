#! /usr/bin/python
# -*- coding:utf-8 -*
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import decomposition
from tempfile import TemporaryFile
import random
from sklearn import neighbors
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

aerolist=pd.read_pickle('aeroports3.0.pkl')



@app.route('/')
def Test_1():
	return render_template('Test_1.html')


@app.route('/result', methods=['POST','GET'])
def result():
	data = pd.read_csv('data_preparrtiFINAL.csv',sep=',')

	date = str(request.args.get('date'))
	orig = str(request.args.get('aerodep'))
	"""
	orig = aerolist.index[aerolist.noms_aeroport==orig]
	if len(orig)>1:
		orig=orig[0]
	else:
		orig=orig.item()
	"""
	arri = str(request.args.get('aeroarr'))
	"""
	arri = aerolist.index[aerolist.noms_aeroport==arri]

	if len(orig)>1:
		arri=arri[0]
	else:
		arri=arri.item()
	"""

	heured = str(request.args.get('heure'))

	data = data[data.ORIGIN==orig]
	data = data[data.heuredep==int(heured)]
	DISTANCE = data.DISTANCE[data.DEST==arri].mean()
	jourvac = data.jour_de_annee[data.FL_DATE==date].mean()
	temps_vol = data.AIR_TIME[data.DEST==arri].mean()



	cible = data.ARR_DELAY
	data = data.drop(data[['ARR_DELAY']],axis=1)

	data = data[['CARRIER_DELAY',
	    'SECURITY_DELAY',
	    'NAS_DELAY',
	    'DISTANCE',
	    'WEATHER_DELAY',
	    'jour_de_annee',
	    'AIR_TIME']]
	dataknn = data[['DISTANCE',
	    'jour_de_annee',
	    'AIR_TIME']]
	X = data.values
	Xknn = dataknn.values
	print('Done')

	std_scale = preprocessing.StandardScaler().fit(X)
	X_scaled = std_scale.transform(X)
	stdknn_scale = preprocessing.StandardScaler().fit(Xknn)
	Xknn_scaled = stdknn_scale.transform(Xknn)
	if len(X_scaled) < 800:
	    data = dataA
	    #Filtre de l'aeroport d'arrivee
	    data = data[data.ORIGIN==data.ORIGIN[a]]
	    cible = data.ARR_DELAY
	    data = data.drop(data[['ARR_DELAY']],axis=1)

	    data = data[['CARRIER_DELAY',
	        'SECURITY_DELAY',
	        'NAS_DELAY',
	        'DISTANCE',
	        'WEATHER_DELAY',
	        'jour_de_annee',
	        'AIR_TIME']]
	    dataknn = data[['DISTANCE',
	        'jour_de_annee',
	        'AIR_TIME']]
	    X = data.values
	    Xknn = dataknn.values
	    print('Done')

	    stdknn_scale = preprocessing.StandardScaler().fit(Xknn)
	    Xknn_scaled = stdknn_scale.transform(Xknn)

	    std_scale = preprocessing.StandardScaler().fit(X)
	    X_scaled = std_scale.transform(X)
	    
	knnnas = neighbors.NearestNeighbors(n_neighbors=3)
	knnnas.fit(Xknn_scaled, np.array(data.NAS_DELAY.values))
	knncar = neighbors.NearestNeighbors(n_neighbors=3)
	knncar.fit(Xknn_scaled, np.array(data.CARRIER_DELAY.values))
	knnsec = neighbors.NearestNeighbors(n_neighbors=3)
	knnsec.fit(Xknn_scaled, np.array(data.SECURITY_DELAY.values))
	knnweath = neighbors.NearestNeighbors(n_neighbors=3)
	knnweath.fit(Xknn_scaled, np.array(data.WEATHER_DELAY.values))

	X_train,X_test,Y_train,Y_test=train_test_split(X_scaled, cible, test_size=0.3)


	alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
	L1 = np.array([1,0.1,0.01,0.001,0.0001,0])

	model2 = ElasticNet()
	grid2 = GridSearchCV(
	    estimator=model2,
	    n_jobs=3, 
	    param_grid={"max_iter": [1, 5, 10],'alpha':alphas,'l1_ratio':L1},
	    scoring='neg_mean_absolute_error',cv=10 )

	grid2.fit(X_train,Y_train)
	Ypredict = grid2.predict(X_test)

	err = abs(Ypredict - Y_test)
	error = sum(err) / len(Y_test)

	R2=sum((Ypredict - np.mean(Y_test))**2) / sum((Y_test - np.mean(Y_test))**2)


	if error<12 and R2>0.45:
	    grid2=pd.read_pickle('picklmod_ssfiltr.pkl')


	d = {'DIST':[DISTANCE],
	    'jourva':[jourvac],
	    'three':[temps_vol]}

	df1 = pd.DataFrame(data=d)
	Xk = stdknn_scale.transform(df1.values)

	def knnpred(Ypredict, arr):
	    Ypredict = Ypredict[1]
	    d = {'one':Ypredict[:,0],
	        'two':Ypredict[:,1],
	        'three':Ypredict[:,2]}
	    df = pd.DataFrame(data=d)
	    df.loc[df.index,'one'] = arr[df.one]
	    df.loc[df.index,'two'] = arr[df.two]
	    df.loc[df.index,'three'] = arr[df.three]
	    df.loc[df.index,'Moy'] = (df.one.values + df.two.values + df.three.values)/3.
	    return df.Moy.values

	Ypredict = knncar.kneighbors(Xk)
	carr=knnpred(Ypredict, np.array(data.CARRIER_DELAY.values))
	carr=carr[0]
	Ypredict = knnsec.kneighbors(Xk)
	secu=knnpred(Ypredict, np.array(data.SECURITY_DELAY.values))
	secu=secu[0]
	Ypredict = knnnas.kneighbors(Xk)
	nas=knnpred(Ypredict, np.array(data.NAS_DELAY.values))
	nas=nas[0]
	Ypredict = knnweath.kneighbors(Xk)
	weath=knnpred(Ypredict, np.array(data.WEATHER_DELAY.values))
	weath=weath[0]

	d = {'1':[carr],'2':[secu],'3':[nas],'4':[DISTANCE],'5':[weath],'6':[jourvac],'7':[temps_vol]}

	df1 = pd.DataFrame(data=d)


	Xfin = std_scale.transform(df1.values)
	Prediction = grid2.predict(Xfin)


	return render_template('result.html', date=int(Prediction[0])) 

if __name__ == '__main__':
	app.run(debug=True)

