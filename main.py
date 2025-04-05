import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


#configuracion de la pagina prediccion inversion tienda de video juego
st.set_page_config(page_title="Predicción de Inversión Tienda de video juego", page_icon=":🎮:", layout="wide")

# Crear punto de entrada def main
def main():
    
    #Cargar imagen videojuego
    st.image("Videojuegos.jpg", width=900)

    #cargar modelo
    filename = 'modelo_reg-tree-RF.pkl' #cargar el modelo de predicción
    model_Tree,model_RF,variables = pickle.load(open(filename, 'rb'))

    #crear el sidbar de variables
    st.sidebar.header("Parámetros del usuario")
    
    
    #crear los campos de entrada para las variables
    def user_input_features():
        edad= st.sidebar.number_input('Edad', min_value=14, max_value=52)
        option = ["'Mass Effect'","'Sim City'","'Crysis'","'Dead Space'", "'Battlefield'", "'KOA: Reckoning'", "'F1'", "'FiFa'"]
        Videojuegos = st.sidebar.selectbox('Videojuegos', option, index=0)
        
        #Entrada Pataformas
        option_plataforma = ["'Play station'","'Xbox'","PC", "Otros"]
        Plataforma = st.sidebar.selectbox('Plataforma', option_plataforma, index=0)


        #Entrada Sexo
        option_sex = ["Mujer", "Hombre"]
        sexo = st.sidebar.selectbox("Sexo",option_sex, index=0)

        #Entrada Consumidor Habitual
        Consumidor_habitual = st.sidebar.checkbox("Consumidor Habitual", value=False)
        
        #Crear diccionario de variables
        data = {
                'Edad': edad,
                'Videojuegos': Videojuegos,
                'Plataforma': Plataforma,
                'Sexo': sexo,
                'Consumidor_habitual': Consumidor_habitual
        }
        
        #Crear un objeto DataFrame de pandas
        data_imput = pd.DataFrame(data, index=[0])
        #st.write(data)
        
        return data_imput
    
    data_imp = user_input_features()
    
    data_preparada =data_imp.copy()
    #st.write(data_preparada)
 
    #Transformar las variables categoricas en variables dummys
    data_preparada = pd.get_dummies(data_preparada, columns=['Videojuegos','Plataforma'], drop_first=False)
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], drop_first=False)
    
    #st.write(data_preparada)
    #Ajustar el dataframe a la forma del modelo Reidexacion de columnas faltantes
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)#Rellenar con 0 las columnas que faltan
    #st.write(data_preparada)
    
#Crear boton de prediccion
    if st.sidebar.button("Predecir"):
        #Realizar la prediccion con el modelo de arbol de decision
        y_pred_tree = model_Tree.predict(data_preparada)

    
        st.success(f"🎮El Cliente Invertira: {y_pred_tree[0]:.1f} dolares")#mostrar la prediccion del modelo Tree
        st.write("prediccion del modelo: 96%")
if __name__ == "__main__":
    main()
