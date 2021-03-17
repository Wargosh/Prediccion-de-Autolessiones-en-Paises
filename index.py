from flask import Flask, render_template
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# para dividir la data para pruebas y test
import sklearn
from sklearn.model_selection import train_test_split
# para observar las metricas de puntaje del algoritmo
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# para usar el algoritmo de regresion lineal
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/action')
def action():
    dataset = pd.read_csv('./data/dataset.csv')
    
    # Eliminar columnas innecesarias para este algoritmo
    dataset = dataset.drop(columns=['age', 'sex', 'gdp_for_year ($)', 'country-year','gdp_per_capita ($)', 'generation', 'HDI for year', 'suicides/100k pop'])
    
    # Cambiando los nombres para un mejor analisis
    dataset = dataset.rename(columns={'year':'Year','suicides_no':'NumSuicides','population':'Population'}) 
    df = pd.DataFrame(data= dataset)
    
    # Seleccionar pais
    _country = "Argentina"

    # Se filtran los datos que sean netamente del país seleccionado
    newData = df[df.country == _country]
    # Eliminar columna de pais porque ya no vuelve a utilizarse
    dataset = dataset.drop(columns=['country'])
    # Se almacena un arreglo con los a;os registrados de dicho pais
    _yearsCountry = newData.Year.sort_values().unique()
    # Se realiza una agrupacion sumando las columnas donde coincida el mismo a;o
    newData = newData.groupby(by=['Year']).sum()
    newData['Year'] = _yearsCountry

    # Se establece en 0 cualquier caracter vacio que pueda entorpercer el algoritmo
    newData = newData.replace(np.nan, "0")
    # se separan los datos en variables para un mejor control
    year = newData['Year'].values
    numSuicides = newData['NumSuicides'].values
    population = newData['Population'].values

    # ************ Algoritmo de Regresion Lineal ************
    reg = LinearRegression()
    # variables independientes [A;o, Poblacion]
    X = np.array([year, population]).T
    # variable dependiente: # de suicidios
    Y = np.array(numSuicides)

    #newDataPlot.set_index('Year', inplace=True)
    subset = ["NumSuicides"]
    s = newData[subset]
    ax = s.plot(marker="o", figsize=(19,10))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.figure.savefig('./static/imgs/my_plot.png')

    print(newData)
    return render_template('page_predict.html')

if __name__ == '__main__':
    app.run(debug = True)