from flask import Flask, render_template, request, redirect, url_for, flash
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

#settings
app.secret_key = 'MLsecretKey'

# global vars
yearPred = 2016

@app.route('/')
def home():
    dataset = pd.read_csv('./data/dataset.csv')
    _countries = dataset.country.sort_values().unique()
    print(_countries)
    return render_template('index.html', Countries = _countries)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/select_country', methods = ['POST'])
def go():
    if request.method == 'POST':
        aux = request.form['_country']

        if (aux != "Elige un pais..." and aux != ""):
            # Seleccionar pais
            _country = aux
            print(_country)
            flash(_country[0])
        else:
            _country = "Argentina" # deja una seleccion por defecto 
            flash('Debes seleccionar un pais')
            return redirect(url_for('home'))

        dataset = pd.read_csv('./data/dataset.csv')
        # Eliminar columnas innecesarias para este algoritmo
        dataset = dataset.drop(columns=['age', 'sex', 'gdp_for_year ($)', 'country-year','gdp_per_capita ($)', 'generation', 'HDI for year', 'suicides/100k pop'])
        # Cambiando los nombres para un mejor analisis
        dataset = dataset.rename(columns={'year':'Year','suicides_no':'NumSuicides','population':'Population'}) 
        df = pd.DataFrame(data= dataset)
        
        # Se filtran los datos que sean netamente del país seleccionado
        newData = df[df.country == _country]
        # Eliminar columna de pais porque ya no vuelve a utilizarse
        newData = newData.drop(columns=['country'])
        # Se almacena un arreglo con los a;os registrados de dicho pais
        _yearsCountry = newData.Year.sort_values().unique()
        # Se realiza una agrupacion sumando las columnas donde coincida el mismo año
        newData = newData.groupby(by=['Year']).sum()
        newData['Year'] = _yearsCountry
        # se transforma el tipo de dataframe a tuplas para reccorrer los datos desde html
        arrData = tuple(zip(newData['Year'], newData['Population'], newData['NumSuicides']))
        print(arrData)
        # Obtener ultima poblacion
        _population = newData.loc[newData.index[-1], "Population"]
        # Obtener ultima poblacion
        lastYear = _yearsCountry[len(_yearsCountry)-1] + 1

        # Generar gráfica de los datos originales
        subset = ["NumSuicides"]
        s = newData[subset]
        ax = s.plot(marker="o", figsize=(32,12))
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.figure.savefig('./static/imgs/my_plot_origin_' + _country + '.png')

        return render_template('viewDataCountry.html', tables = arrData, country = _country, lastPopulation = _population, yearRecommended = lastYear)

@app.route('/prediction', methods = ['POST'])
def prediction():
    if request.method == 'POST':
        # Seleccionar pais
        _country = request.form['_country']
        yearPred = int(request.form['_year'])
    else:
        return redirect(url_for('home'))

    dataset = pd.read_csv('./data/dataset.csv')
    
    # Eliminar columnas innecesarias para este algoritmo
    dataset = dataset.drop(columns=['age', 'sex', 'gdp_for_year ($)', 'country-year','gdp_per_capita ($)', 'generation', 'HDI for year', 'suicides/100k pop'])
    
    # Cambiando los nombres para un mejor analisis
    dataset = dataset.rename(columns={'year':'Year','suicides_no':'NumSuicides','population':'Population'}) 
    df = pd.DataFrame(data= dataset)
    
    # Se filtran los datos que sean netamente del país seleccionado
    newData = df[df.country == _country]
    # Eliminar columna de pais porque ya no vuelve a utilizarse
    newData = newData.drop(columns=['country'])
    # Se almacena un arreglo con los a;os registrados de dicho pais
    _yearsCountry = newData.Year.sort_values().unique()
    # Se realiza una agrupacion sumando las columnas donde coincida el mismo año
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
    # variables independientes [Año, Población]
    X = np.array([year, population]).T
    # variable dependiente: # de suicidios
    Y = np.array(numSuicides)
    # Entrenar el modelo
    reg = reg.fit(X, Y)
    Y_predict = reg.predict(X)
    # Obtener puntaje de varianza (el puntaje es mas eficiente es 1.0)
    variance = r2_score(Y, Y_predict)

    # Datos de entrada
    _year = yearPred
    _population = newData.loc[newData.index[-1], "Population"]

    # Realizar predicción
    res_pred = reg.predict([[_year, _population]])

    # Agregar la predicción a los datos originales
    newDataPlot = newData.append(
    pd.Series([res_pred[0], _population, _year], index=newData.columns), ignore_index=True)
    # establecer la columna de año como un campo de tiempo para los plots 
    newDataPlot.set_index('Year', inplace=True)

    # Generar gráfica de los datos con la predicción
    subset = ["NumSuicides"]
    ax = newDataPlot[subset].plot(marker="o", figsize=(19,10))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.plot(_yearsCountry, Y_predict, color='blue')
    x_real = _yearsCountry
    y_real = Y
    plt.plot(x_real, y_real, color='green')

    s = newDataPlot[subset]
    ax = s.plot(marker="o", figsize=(28,12))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.plot(_yearsCountry, Y_predict, color='blue')
    x_real = _yearsCountry
    y_real = Y
    plt.plot(x_real, y_real, color='green')
    ax.figure.savefig('./static/imgs/my_plot_predict_' + _country + '.png')

    print(variance)
    return render_template('page_predict.html', country = _country, year = _year, variance = variance)

if __name__ == '__main__':
    app.run(debug = True)