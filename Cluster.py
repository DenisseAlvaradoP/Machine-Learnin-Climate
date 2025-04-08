import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('linear_regression_model_with_clusters.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define custom styles
app.layout = html.Div(style={'backgroundColor': '#000000', 'color': '#FFFFFF', 'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1('Average Daily Temperature Prediction', style={'color': '#39FF14', 'paddingTop': '20px'}),

    html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '60%', 'margin': '0 auto', 'marginBottom': '20px'}, children=[
        html.Label('Select Month:', style={'color': '#FFFFFF'}),
        dcc.Dropdown(
            id='month-dropdown',
            options=[
                {'label': 'January', 'value': 'January'},
                {'label': 'February', 'value': 'February'},
                {'label': 'March', 'value': 'March'},
                {'label': 'April', 'value': 'April'},
                {'label': 'May', 'value': 'May'},
                {'label': 'June', 'value': 'June'},
                {'label': 'July', 'value': 'July'},
                {'label': 'August', 'value': 'August'},
                {'label': 'September', 'value': 'September'},
                {'label': 'October', 'value': 'October'},
                {'label': 'November', 'value': 'November'},
                {'label': 'December', 'value': 'December'}
            ],
            value='January',  # Default value
            style={'width': '50%'}
        )
    ]),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'padding': '20px'}, children=[
        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/wind.png', style={'width': '50px'}),
            html.Label('Wind Speed (m/s)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='wind_speed_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='wind_speed', min=0, max=50, step=0.1, value=2.0, marks={i: str(i) for i in range(0, 51, 5)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/precipitation.png', style={'width': '50px'}),
            html.Label('Precipitation Level (mm)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='precipitation_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='precipitation', min=0, max=200, step=0.1, value=0.0, marks={i: str(i) for i in range(0, 201, 20)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/sun.png', style={'width': '50px'}),
            html.Label('Sun Duration (hours)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='sun_duration_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='sun_duration', min=0, max=18, step=0.1, value=8.0, marks={i: str(i) for i in range(0, 19, 2)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/snow.png', style={'width': '50px'}),
            html.Label('Snow Height (cm)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='snow_height_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='snow_height', min=0, max=100, step=0.1, value=0.0, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/cloud.png', style={'width': '50px'}),
            html.Label('Cloud Cover (octaves)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='cloud_cover_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='cloud_cover', min=0, max=8, step=1, value=4, marks={i: str(i) for i in range(0, 9, 1)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/vapor.png', style={'width': '50px'}),
            html.Label('Vapor Pressure (hPa)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='vapor_pressure_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='vapor_pressure', min=0, max=30, step=0.1, value=10.0, marks={i: str(i) for i in range(0, 31, 5)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/pressure.png', style={'width': '50px'}),
            html.Label('Atmospheric Pressure (hPa)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='atmospheric_pressure_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='atmospheric_pressure', min=1000, max=1060, step=0.1, value=1013.25, marks={i: str(i) for i in range(1000, 1061, 10)}, tooltip={"placement": "bottom"})
        ]),

        html.Div(style={'backgroundColor': '#1c1c1c', 'padding': '20px', 'borderRadius': '10px', 'width': '45%', 'marginBottom': '20px'}, children=[
            html.Img(src='/assets/humidity.png', style={'width': '50px'}),
            html.Label('Relative Humidity (%)', style={'color': '#FFFFFF'}),
            dcc.RadioItems(id='humidity_cluster', options=[
                {'label': 'Low', 'value': 'low'},
                {'label': 'Medium', 'value': 'medium'},
                {'label': 'High', 'value': 'high'}
            ], value='low', labelStyle={'display': 'inline-block', 'color': '#39FF14'}),
            dcc.Slider(id='humidity', min=0, max=100, step=0.1, value=50.0, marks={i: str(i) for i in range(0, 101, 10)}, tooltip={"placement": "bottom"})
        ])
    ]),

    html.Button('Adjust Cluster Parameters', id='adjust_cluster_button', n_clicks=0, style={'backgroundColor': '#39FF14', 'color': '#000000', 'padding': '10px 20px', 'fontSize': '16px', 'marginTop': '20px'}),

    html.Button('Predict Temperature', id='predict-button', n_clicks=0, style={'backgroundColor': '#39FF14', 'color': '#000000', 'padding': '10px 20px', 'fontSize': '16px', 'marginTop': '20px'}),

    html.H2(id='prediction-output', style={'color': '#FFFFFF', 'paddingTop': '20px'})
])

# Define the callback to adjust the cluster parameters
@app.callback(
    Output('wind_speed_cluster', 'value'),
    Output('precipitation_cluster', 'value'),
    Output('sun_duration_cluster', 'value'),
    Output('snow_height_cluster', 'value'),
    Output('cloud_cover_cluster', 'value'),
    Output('vapor_pressure_cluster', 'value'),
    Output('atmospheric_pressure_cluster', 'value'),
    Output('humidity_cluster', 'value'),
    Input('adjust_cluster_button', 'n_clicks'),
    State('wind_speed', 'value'),
    State('precipitation', 'value'),
    State('sun_duration', 'value'),
    State('snow_height', 'value'),
    State('cloud_cover', 'value'),
    State('vapor_pressure', 'value'),
    State('atmospheric_pressure', 'value'),
    State('humidity', 'value')
)
def adjust_cluster_parameters(n_clicks, wind_speed, precipitation, sun_duration, snow_height, cloud_cover, vapor_pressure, atmospheric_pressure, humidity):
    # Logic to adjust the cluster parameters based on input values
    wind_speed_cluster = 'low' if wind_speed < 10 else 'medium' if wind_speed < 20 else 'high'
    precipitation_cluster = 'low' if precipitation < 50 else 'medium' if precipitation < 100 else 'high'
    sun_duration_cluster = 'low' if sun_duration < 6 else 'medium' if sun_duration < 12 else 'high'
    snow_height_cluster = 'low' if snow_height < 20 else 'medium' if snow_height < 50 else 'high'
    cloud_cover_cluster = 'low' if cloud_cover < 3 else 'medium' if cloud_cover < 6 else 'high'
    vapor_pressure_cluster = 'low' if vapor_pressure < 10 else 'medium' if vapor_pressure < 20 else 'high'
    atmospheric_pressure_cluster = 'low' if atmospheric_pressure < 1010 else 'medium' if atmospheric_pressure < 1030 else 'high'
    humidity_cluster = 'low' if humidity < 30 else 'medium' if humidity < 70 else 'high'
    
    return (
        wind_speed_cluster, precipitation_cluster, sun_duration_cluster, snow_height_cluster, cloud_cover_cluster, 
        vapor_pressure_cluster, atmospheric_pressure_cluster, humidity_cluster
    )

# Define the callback to predict the temperature
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('month-dropdown', 'value'),
    State('wind_speed', 'value'),
    State('precipitation', 'value'),
    State('sun_duration', 'value'),
    State('snow_height', 'value'),
    State('cloud_cover', 'value'),
    State('vapor_pressure', 'value'),
    State('atmospheric_pressure', 'value'),
    State('humidity', 'value'),
    State('wind_speed_cluster', 'value'),
    State('precipitation_cluster', 'value'),
    State('sun_duration_cluster', 'value'),
    State('snow_height_cluster', 'value'),
    State('cloud_cover_cluster', 'value'),
    State('vapor_pressure_cluster', 'value'),
    State('atmospheric_pressure_cluster', 'value'),
    State('humidity_cluster', 'value')
)
def predict_temperature(n_clicks, month, wind_speed, precipitation, sun_duration, snow_height, cloud_cover, vapor_pressure, atmospheric_pressure, humidity,
                        wind_speed_cluster, precipitation_cluster, sun_duration_cluster, snow_height_cluster, cloud_cover_cluster,
                        vapor_pressure_cluster, atmospheric_pressure_cluster, humidity_cluster):
    if n_clicks > 0:
        # Map the month name to the month number
        month_mapping = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_number = month_mapping.get(month, 1)  # Default to January if month is not found

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Month': [month_number],
            'Wind Speed (m/s)': [wind_speed],
            'Precipitation Level (mm)': [precipitation],
            'Sun Duration (hours)': [sun_duration],
            'Snow Height (cm)': [snow_height],
            'Cloud Cover (octaves)': [cloud_cover],
            'Vapor Pressure (hPa)': [vapor_pressure],
            'Atmospheric Pressure (hPa)': [atmospheric_pressure],
            'Relative Humidity (%)': [humidity]
        })

        # Use the model to make a prediction
        prediction = model.predict(input_data)[0]

        return f'The predicted average daily temperature is {prediction:.2f} °C'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)