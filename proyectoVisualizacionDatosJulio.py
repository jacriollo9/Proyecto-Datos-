import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import numpy as np

# Cargar el dataset filtrado
df2 = pd.read_csv('C:/Users/josep/Downloads/New Folder/FilteredDataset.csv', low_memory=False)

# Aplicar la conversión a las columnas 'Value' y 'Wage'
def value_to_numeric(value):
    if 'M' in value:
        return float(value.replace('€', '').replace('M', '')) * 1e6
    if 'K' in value:
        return float(value.replace('€', '').replace('K', '')) * 1e3
    return float(value.replace('€', ''))

df2['Value'] = df2['Value'].apply(value_to_numeric)
df2['Wage'] = df2['Wage'].apply(value_to_numeric)

# Traducción de habilidades al español
skill_columns_english = [
    'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 
    'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 
    'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 
    'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 
    'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys'
]

skill_columns_spanish = [
    'Aceleración', 'Agresividad', 'Agilidad', 'Equilibrio', 'Control del balón', 'Compostura', 'Centros', 
    'Efecto', 'Regate', 'Definición', 'Precisión de tiros libres', 'Voleo del portero', 'Manejo del portero', 
    'Saque del portero', 'Posicionamiento del portero', 'Reflejos del portero', 'Precisión de cabezazo', 
    'Intercepciones', 'Salto', 'Pase largo', 'Tiros largos', 'Marcaje', 'Penaltis', 'Posicionamiento', 
    'Reacciones', 'Pase corto', 'Potencia de tiro', 'Entrada deslizante', 'Velocidad de sprint', 'Resistencia', 
    'Entrada normal', 'Fuerza', 'Visión', 'Voleas'
]

skill_translation = dict(zip(skill_columns_english, skill_columns_spanish))

# Convertir las columnas de habilidades a números
for column in skill_columns_english:
    df2[column] = df2[column].str.extract(r'(\d+)').astype(float)

# Traducción de posiciones al español
position_translation = {
    'ST': 'Delantero Centro', 
    'LW': 'Extremo Izquierdo', 
    'RW': 'Extremo Derecho', 
    'GK': 'Portero', 
    'CB': 'Defensa Central', 
    'LB': 'Lateral Izquierdo', 
    'RB': 'Lateral Derecho', 
    'CM': 'Centrocampista', 
    'CAM': 'Centrocampista Ofensivo', 
    'CDM': 'Centrocampista Defensivo', 
    'LM': 'Centrocampista Izquierdo', 
    'RM': 'Centrocampista Derecho'
}

df2['Preferred Positions'] = df2['Preferred Positions'].apply(lambda x: ', '.join([position_translation.get(pos, pos) for pos in x.split()]))

# Configurar la página de Streamlit
st.set_page_config(page_title='Visualización de Datos de Jugadores de Fútbol', layout='wide')

# Título principal
st.title("Visualización de Datos de Jugadores de Fútbol")
st.markdown("""
    Bienvenido a la visualización de datos de jugadores de fútbol. Aquí podrás explorar estadísticas detalladas 
    de los jugadores, comparar sus habilidades y realizar filtros avanzados para analizar la información de manera más precisa.
""")

# Resumen Global
st.subheader("Resumen Global")
col1, col2, col3 = st.columns(3)
col1.metric("Número total de jugadores", len(df2))
col2.metric("Distribución por país", df2['Nationality'].nunique())
col3.metric("Distribución por categoría", df2['Preferred Positions'].nunique())

# Barra Lateral de Navegación
st.sidebar.title("Navegación")
section = st.sidebar.radio("Seleccione una sección:", ["Visualización general por países", "Estadísticas de habilidades por jugador", "Filtrado de jugadores más complejos", "Estadísticas adicionales"])

# Visualización general por países
if section == "Visualización general por países":
    st.subheader("Análisis Detallado")

    # Barra deslizante para ajustar el valor de la habilidad
    skill_threshold = st.slider("Seleccione el valor mínimo de la habilidad", 0, 100, 75)

    # Crear gráficos de barras horizontales para habilidades con valores mayores al umbral seleccionado
    rows = 3
    cols = 2

    for i in range(rows):
        columns = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < len(skill_columns_spanish[:6]):  # Solo los primeros 6 skills para la visualización
                skill = skill_columns_english[idx]
                skill_spanish = skill_translation[skill]
                filtered_df = df2[df2[skill] > skill_threshold]
                top_countries = filtered_df['Nationality'].value_counts().nlargest(6).reset_index()
                top_countries.columns = ['Nationality', 'Player Count']

                if not top_countries.empty:
                    fig = px.bar(top_countries, y='Nationality', x='Player Count', orientation='h', 
                                 title=f'Países con más jugadores con {skill_spanish} mayor {skill_threshold}',
                                 labels={'Player Count': 'Número de jugadores', 'Nationality': 'País'},
                                 color='Player Count', color_continuous_scale='Blues')

                    fig.update_layout(
                        template='plotly_white',
                        font=dict(family='Roboto, Arial', size=12)
                    )

                    columns[j].plotly_chart(fig, use_container_width=True)
                else:
                    columns[j].write(f"No hay jugadores con {skill_spanish} > {skill_threshold}.")

elif section == "Estadísticas de habilidades por jugador":
    st.subheader("Análisis Detallado")
    st.markdown("Seleccione las posiciones y habilidades para comparar el desempeño de los jugadores.")

    # Filtros de Dropdown
    positions = st.multiselect("Filtrar por posición", options=df2['Preferred Positions'].unique(), key='position-filter')
    skill1 = st.selectbox("Seleccionar la primera habilidad", options=skill_columns_spanish, key='skill1-filter')
    skill2 = st.selectbox("Seleccionar la segunda habilidad", options=skill_columns_spanish, key='skill2-filter')

    # Aplicar filtros de posición
    dff = df2.copy()
    if positions:
        dff = dff[dff['Preferred Positions'].apply(lambda x: any(pos in x for pos in positions))]

    # Traducir habilidades seleccionadas al inglés para el análisis
    skill1_eng = [k for k, v in skill_translation.items() if v == skill1][0]
    skill2_eng = [k for k, v in skill_translation.items() if v == skill2][0]

    # Crear gráfico de dispersión para comparar las dos habilidades seleccionadas y colorear por nacionalidad
    if skill1 and skill2:
        fig = px.scatter(dff, x=skill1_eng, y=skill2_eng, color='Nationality',
                         hover_data={'Name': True, 'Nationality': True, 'Value': True},
                         title=f'Comparar {skill1} con {skill2}',
                         labels={skill1_eng: skill1, skill2_eng: skill2},
                         color_discrete_sequence=px.colors.qualitative.Safe)
        
        fig.update_traces(marker=dict(size=12, 
                                      line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        
        fig.update_layout(
            template='plotly_white',
            font=dict(family='Roboto, Arial', size=12),
            legend_title_text='Nacionalidad'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Seleccione dos habilidades para comparar.")

# Filtrado de jugadores más complejos
elif section == "Filtrado de jugadores más complejos":
    st.subheader("Filtrado Avanzado")
    st.markdown("Utilice los filtros a continuación para encontrar jugadores que se ajusten a criterios específicos.")

    # Filtrar por rango de edades
    min_age, max_age = st.slider("Rango de Edad", int(df2['Age'].min()), int(df2['Age'].max()), (20, 30))

    # Filtrar por país de origen
    countries = st.multiselect("Seleccionar país(es) de origen", options=df2['Nationality'].unique(), key='country-filter')

    # Filtrar por valor de jugador
    min_value, max_value = st.slider("Rango de Valor del Jugador", int(df2['Value'].min()), int(df2['Value'].max()), (0, int(df2['Value'].max() / 2)))

    # Filtrar por posición
    positions = st.multiselect("Filtrar por posición", options=df2['Preferred Positions'].unique(), key='position-filter-more')

    # Filtrar por una habilidad específica
    skill = st.selectbox("Seleccionar habilidad", options=skill_columns_spanish, key='skill-filter-more')

    # Traducir la habilidad seleccionada al inglés
    skill_eng = [k for k, v in skill_translation.items() if v == skill][0]

    # Aplicar filtros
    dff = df2.copy()
    dff = dff[(dff['Age'] >= min_age) & (dff['Age'] <= max_age)]
    if countries:
        dff = dff[dff['Nationality'].isin(countries)]
    dff = dff[(dff['Value'] >= min_value) & (dff['Value'] <= max_value)]
    if positions:
        dff = dff[dff['Preferred Positions'].apply(lambda x: any(pos in x for pos in positions))]
    if skill:
        dff = dff[dff[skill_eng] > 0]

    # Mostrar resultados filtrados
    st.write(f"Se encontraron {len(dff)} jugadores con los criterios seleccionados.")

    # Mostrar tabla de resultados filtrados
    st.dataframe(dff)

    # Crear gráfico de dispersión para comparar las habilidades y colorear por nacionalidad
    if skill:
        fig = px.scatter(dff, x='Age', y=skill_eng, color='Nationality',
                         hover_data={'Name': True, 'Nationality': True, 'Value': True},
                         title=f'Comparar Edad con {skill}',
                         labels={'Age': 'Edad', skill_eng: skill},
                         color_discrete_sequence=px.colors.qualitative.Safe)
        
        fig.update_traces(marker=dict(size=12, 
                                      line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        
        fig.update_layout(
            template='plotly_white',
            font=dict(family='Roboto, Arial', size=12),
            legend_title_text='Nacionalidad'
        )

        st.plotly_chart(fig, use_container_width=True)

elif section == "Estadísticas adicionales":
    st.subheader("Estadísticas Adicionales")
    st.markdown("Explore estadísticas adicionales sobre los jugadores en el dataset.")

    # Mapa de calor de la distribución de jugadores por nacionalidad
    st.subheader("Mapa de Calor de la Distribución de Jugadores por Nacionalidad")
    nationality_counts = df2['Nationality'].value_counts().reset_index()
    nationality_counts.columns = ['Nacionalidad', 'Número de Jugadores']
    fig_heatmap_nationality = px.density_heatmap(nationality_counts, x='Nacionalidad', y='Número de Jugadores',
                                                 title='Mapa de Calor de la Distribución de Jugadores por Nacionalidad',
                                                 labels={'Número de Jugadores': 'Número de Jugadores', 'Nacionalidad': 'Nacionalidad'})
    st.plotly_chart(fig_heatmap_nationality, use_container_width=True)

    # Gráfico de barras para la distribución de valores de jugadores por nacionalidad
    st.subheader("Distribución de Valores de Jugadores por Nacionalidad")
    fig_box_value = px.box(df2, x='Nationality', y='Value',
                           title='Distribución de Valores de Jugadores por Nacionalidad',
                           labels={'Value': 'Valor del Jugador', 'Nationality': 'Nacionalidad'})
    st.plotly_chart(fig_box_value, use_container_width=True)

    # Histograma de edades de los jugadores
    st.subheader("Histograma de Edades de los Jugadores")
    fig_hist_age = px.histogram(df2, x='Age', nbins=20, title='Distribución de Edades de los Jugadores',
                                labels={'Age': 'Edad', 'count': 'Número de Jugadores'})
    st.plotly_chart(fig_hist_age, use_container_width=True)

    # Gráfico 3D de dispersión para comparar Valor vs Edad vs Habilidad seleccionada
    if skill_columns_spanish:
        st.subheader(f"Comparación 3D de Valor, Edad y {skill_columns_spanish}")
        fig_scatter_3d = px.scatter_3d(df2, x='Age', y='Value', z=skill_columns_spanish,
                                       color='Nationality', size='Value',
                                       hover_name='Name', opacity=0.7,
                                       title=f'Comparación 3D de Valor, Edad y {skill_columns_spanish}',
                                       labels={'Age': 'Edad', 'Value': 'Valor del Jugador', 'Nationality': 'Nacionalidad'})
        st.plotly_chart(fig_scatter_3d, use_container_width=True)

# Detalles del jugador seleccionado
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

click_data = st.session_state.selected_player

if click_data:
    player_name = click_data['points'][0]['hovertext']
    player_info = df2[df2['Name'] == player_name].iloc[0]
    st.write(f"### {player_name}")
    st.write(f"Nacionalidad: {player_info['Nationality']}")
    st.write(f"Valor: €{player_info['Value']:,.0f}")

# Mensaje para indicar cómo ejecutar la aplicación
st.write("Para ver esta aplicación en el navegador, ejecuta el siguiente comando en tu terminal:")
st.code("streamlit run c:/Users/josep/Desktop/proyectoVisualizacionDatosJulio.py")
