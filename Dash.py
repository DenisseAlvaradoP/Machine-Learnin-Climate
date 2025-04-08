#Dash
cluster_ranges = {
    'low': {
        'Wind Speed (m/s)': (0, 10),
        'Precipitation Level (mm)': (0, 10),
        'Sun Duration (hours)': (0, 6),
        'Snow Height (cm)': (0, 20),
        'Cloud Cover (octaves)': (0, 4),
        'Vapor Pressure (hPa)': (0, 10),
        'Atmospheric Pressure (hPa)': (1000, 1020),
        'Relative Humidity (%)': (0, 50)
    },
    'medium': {
        'Wind Speed (m/s)': (10, 20),
        'Precipitation Level (mm)': (10, 50),
        'Sun Duration (hours)': (6, 12),
        'Snow Height (cm)': (20, 50),
        'Cloud Cover (octaves)': (4, 6),
        'Vapor Pressure (hPa)': (10, 20),
        'Atmospheric Pressure (hPa)': (1020, 1040),
        'Relative Humidity (%)': (50, 75)
    },
    'high': {
        'Wind Speed (m/s)': (20, 50),
        'Precipitation Level (mm)': (50, 200),
        'Sun Duration (hours)': (12, 18),
        'Snow Height (cm)': (50, 100),
        'Cloud Cover (octaves)': (6, 8),
        'Vapor Pressure (hPa)': (20, 30),
        'Atmospheric Pressure (hPa)': (1040, 1060),
        'Relative Humidity (%)': (75, 100)
    }
}

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('linear_regression_model_with_clusters.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Average Daily Temperature Prediction'),

    html.Label('Month (1-12)'),
    dcc.Slider(id='month', min=1, max=12, step=1, value=1, marks={i: str(i) for i in range(1, 13)}),

    html.Label('Wind Speed (m/s)'),
    dcc.RadioItems(id='wind_speed_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='wind_speed', min=0, max=50, step=0.1, value=2.0),

    html.Label('Precipitation Level (mm)'),
    dcc.RadioItems(id='precipitation_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='precipitation', min=0, max=200, step=0.1, value=0.0),

    html.Label('Sun Duration (hours)'),
    dcc.RadioItems(id='sun_duration_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='sun_duration', min=0, max=18, step=0.1, value=8.0),

    html.Label('Snow Height (cm)'),
    dcc.RadioItems(id='snow_height_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='snow_height', min=0, max=100, step=0.1, value=0.0),

    html.Label('Cloud Cover (octaves)'),
    dcc.RadioItems(id='cloud_cover_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='cloud_cover', min=0, max=8, step=1, value=4),

    html.Label('Vapor Pressure (hPa)'),
    dcc.RadioItems(id='vapor_pressure_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='vapor_pressure', min=0, max=30, step=0.1, value=10.0),

    html.Label('Atmospheric Pressure (hPa)'),
    dcc.RadioItems(id='atmospheric_pressure_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='atmospheric_pressure', min=1000, max=1060, step=0.1, value=1013.25),

    html.Label('Relative Humidity (%)'),
    dcc.RadioItems(id='humidity_cluster', options=[
        {'label': 'Low', 'value': 'low'},
        {'label': 'Medium', 'value': 'medium'},
        {'label': 'High', 'value': 'high'}
    ], value='low'),
    dcc.Slider(id='humidity', min=0, max=100, step=0.1, value=50.0),

    html.Button('Adjust Cluster Parameters', id='adjust_cluster_button', n_clicks=0),

    html.Button('Predict Temperature', id='predict-button', n_clicks=0),

    html.H2('The predicted average temperature is:'),
    html.Div(id='prediction-output')
])

@app.callback(
    [Output('wind_speed', 'min'),
     Output('wind_speed', 'max'),
     Output('wind_speed', 'value'),
     Output('wind_speed', 'marks'),
     Output('precipitation', 'min'),
     Output('precipitation', 'max'),
     Output('precipitation', 'value'),
     Output('precipitation', 'marks'),
     Output('sun_duration', 'min'),
     Output('sun_duration', 'max'),
     Output('sun_duration', 'value'),
     Output('sun_duration', 'marks'),
     Output('snow_height', 'min'),
     Output('snow_height', 'max'),
     Output('snow_height', 'value'),
     Output('snow_height', 'marks'),
     Output('cloud_cover', 'min'),
     Output('cloud_cover', 'max'),
     Output('cloud_cover', 'value'),
     Output('cloud_cover', 'marks'),
     Output('vapor_pressure', 'min'),
     Output('vapor_pressure', 'max'),
     Output('vapor_pressure', 'value'),
     Output('vapor_pressure', 'marks'),
     Output('atmospheric_pressure', 'min'),
     Output('atmospheric_pressure', 'max'),
     Output('atmospheric_pressure', 'value'),
     Output('atmospheric_pressure', 'marks'),
     Output('humidity', 'min'),
     Output('humidity', 'max'),
     Output('humidity', 'value'),
     Output('humidity', 'marks')],
    [Input('adjust_cluster_button', 'n_clicks')],
    [State('wind_speed_cluster', 'value'),
     State('precipitation_cluster', 'value'),
     State('sun_duration_cluster', 'value'),
     State('snow_height_cluster', 'value'),
     State('cloud_cover_cluster', 'value'),
     State('vapor_pressure_cluster', 'value'),
     State('atmospheric_pressure_cluster', 'value'),
     State('humidity_cluster', 'value')]
)
def adjust_cluster_parameters(n_clicks, wind_speed_cluster, precipitation_cluster, sun_duration_cluster,
                              snow_height_cluster, cloud_cover_cluster, vapor_pressure_cluster, 
                              atmospheric_pressure_cluster, humidity_cluster):
    if n_clicks > 0:
        wind_speed_range = cluster_ranges[wind_speed_cluster]
        precipitation_range = cluster_ranges[precipitation_cluster]
        sun_duration_range = cluster_ranges[sun_duration_cluster]
        snow_height_range = cluster_ranges[snow_height_cluster]
        cloud_cover_range = cluster_ranges[cloud_cover_cluster]
        vapor_pressure_range = cluster_ranges[vapor_pressure_cluster]
        atmospheric_pressure_range =cluster_ranges[atmospheric_pressure_cluster]
        humidity_range = cluster_ranges[humidity_cluster]

        return (
            wind_speed_range['Wind Speed (m/s)'][0], wind_speed_range['Wind Speed (m/s)'][1], wind_speed_range['Wind Speed (m/s)'][0],
            {wind_speed_range['Wind Speed (m/s)'][0]: str(wind_speed_range['Wind Speed (m/s)'][0]), wind_speed_range['Wind Speed (m/s)'][1]: str(wind_speed_range['Wind Speed (m/s)'][1])},
            precipitation_range['Precipitation Level (mm)'][0], precipitation_range['Precipitation Level (mm)'][1], precipitation_range['Precipitation Level (mm)'][0],
            {precipitation_range['Precipitation Level (mm)'][0]: str(precipitation_range['Precipitation Level (mm)'][0]), precipitation_range['Precipitation Level (mm)'][1]: str(precipitation_range['Precipitation Level (mm)'][1])},
            sun_duration_range['Sun Duration (hours)'][0], sun_duration_range['Sun Duration (hours)'][1], sun_duration_range['Sun Duration (hours)'][0],
            {sun_duration_range['Sun Duration (hours)'][0]: str(sun_duration_range['Sun Duration (hours)'][0]), sun_duration_range['Sun Duration (hours)'][1]: str(sun_duration_range['Sun Duration (hours)'][1])},
            snow_height_range['Snow Height (cm)'][0], snow_height_range['Snow Height (cm)'][1], snow_height_range['Snow Height (cm)'][0],
            {snow_height_range['Snow Height (cm)'][0]: str(snow_height_range['Snow Height (cm)'][0]), snow_height_range['Snow Height (cm)'][1]: str(snow_height_range['Snow Height (cm)'][1])},
            cloud_cover_range['Cloud Cover (octaves)'][0], cloud_cover_range['Cloud Cover (octaves)'][1], cloud_cover_range['Cloud Cover (octaves)'][0],
            {cloud_cover_range['Cloud Cover (octaves)'][0]: str(cloud_cover_range['Cloud Cover (octaves)'][0]), cloud_cover_range['Cloud Cover (octaves)'][1]: str(cloud_cover_range['Cloud Cover (octaves)'][1])},
            vapor_pressure_range['Vapor Pressure (hPa)'][0], vapor_pressure_range['Vapor Pressure (hPa)'][1], vapor_pressure_range['Vapor Pressure (hPa)'][0],
            {vapor_pressure_range['Vapor Pressure (hPa)'][0]: str(vapor_pressure_range['Vapor Pressure (hPa)'][0]), vapor_pressure_range['Vapor Pressure (hPa)'][1]: str(vapor_pressure_range['Vapor Pressure (hPa)'][1])},
            atmospheric_pressure_range['Atmospheric Pressure (hPa)'][0], atmospheric_pressure_range['Atmospheric Pressure (hPa)'][1], atmospheric_pressure_range['Atmospheric Pressure (hPa)'][0],
            {atmospheric_pressure_range['Atmospheric Pressure (hPa)'][0]: str(atmospheric_pressure_range['Atmospheric Pressure (hPa)'][0]), atmospheric_pressure_range['Atmospheric Pressure (hPa)'][1]: str(atmospheric_pressure_range['Atmospheric Pressure (hPa)'][1])},
            humidity_range['Relative Humidity (%)'][0], humidity_range['Relative Humidity (%)'][1], humidity_range['Relative Humidity (%)'][0],
            {humidity_range['Relative Humidity (%)'][0]: str(humidity_range['Relative Humidity (%)'][0]), humidity_range['Relative Humidity (%)'][1]: str(humidity_range['Relative Humidity (%)'][1])}
        )
    return (
        0, 50, 2.0, {0: '0', 50: '50'},
        0, 200, 0.0, {0: '0', 200: '200'},
        0, 18, 8.0, {0: '0', 18: '18'},
        0, 100, 0.0, {0: '0', 100: '100'},
        0, 8, 4, {0: '0', 8: '8'},
        0, 30, 10.0, {0: '0', 30: '30'},
        1000, 1060, 1013.25, {1000: '1000', 1060: '1060'},
        0, 100, 50.0, {0: '0', 100: '100'}
    )

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('month', 'value'),
     State('wind_speed', 'value'),
     State('precipitation', 'value'),
     State('sun_duration', 'value'),
     State('snow_height', 'value'),
     State('cloud_cover', 'value'),
     State('vapor_pressure', 'value'),
     State('atmospheric_pressure', 'value'),
     State('humidity', 'value')]
)
def predict_temperature(n_clicks, month, wind_speed, precipitation, sun_duration,
                        snow_height, cloud_cover, vapor_pressure, atmospheric_pressure, humidity):
    if n_clicks > 0:
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'Month': [month],
            'Wind Speed (m/s)': [wind_speed],
            'Precipitation Level (mm)': [precipitation],
            'Sun Duration (hours)': [sun_duration],
            'Snow Height (cm)': [snow_height],
            'Cloud Cover (octaves)': [cloud_cover],
            'Vapor Pressure (hPa)': [vapor_pressure],
            'Atmospheric Pressure (hPa)': [atmospheric_pressure],
            'Relative Humidity (%)': [humidity]
        })

        # Predict the temperature
        prediction = model.predict(input_data)[0]

        # Determine the temperature cluster
        if prediction < 10:
            temperature_cluster = 'Low'
        elif 10 <= prediction < 20:
            temperature_cluster = 'Medium'
        else:
            temperature_cluster = 'High'

        # Return the prediction and temperature cluster
        return f'The predicted average temperature is: {prediction:.2f} °C, which belongs to the {temperature_cluster} cluster'
    return ''

@app.callback(
    Output('snow_height', 'disabled'),
    [Input('month', 'value')]
)
def disable_snow_height(month):
    # Disable snow height slider for non-winter months (considering winter months as December, January, and February)
    if month in [12, 1, 2]:
        return False
    return True

if __name__ == '_main_':
    app.run_server(debug=True)