import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
import json

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Game Automation Monitoring Dashboard'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='requests-graph')
        ], className='six columns'),
        html.Div([
            dcc.Graph(id='errors-graph')
        ], className='six columns')
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(id='processing-time-graph')
        ], className='six columns'),
        html.Div([
            dcc.Graph(id='game-state-graph')
        ], className='six columns')
    ], className='row')
])

@app.callback(Output('requests-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_requests_graph(n):
    response = requests.get('http://localhost:8000/metrics')
    metrics = response.text.split('\n')
    
    request_counts = {}
    for metric in metrics:
        if metric.startswith('game_automation_requests_total'):
            parts = metric.split()
            request_type = parts[0].split('{')[1].split('}')[0].split('=')[1].strip('"')
            count = float(parts[1])
            request_counts[request_type] = count

    return {
        'data': [go.Bar(x=list(request_counts.keys()), y=list(request_counts.values()))],
        'layout': go.Layout(title='Total Requests by Type')
    }

@app.callback(Output('errors-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_errors_graph(n):
    response = requests.get('http://localhost:8000/metrics')
    metrics = response.text.split('\n')
    
    error_counts = {}
    for metric in metrics:
        if metric.startswith('game_automation_errors_total'):
            parts = metric.split()
            error_type = parts[0].split('{')[1].split('}')[0].split('=')[1].strip('"')
            count = float(parts[1])
            error_counts[error_type] = count

    return {
        'data': [go.Bar(x=list(error_counts.keys()), y=list(error_counts.values()))],
        'layout': go.Layout(title='Total Errors by Type')
    }

@app.callback(Output('processing-time-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_processing_time_graph(n):
    response = requests.get('http://localhost:8000/metrics')
    metrics = response.text.split('\n')
    
    processing_times = {}
    for metric in metrics:
        if metric.startswith('game_automation_processing_seconds_sum'):
            parts = metric.split()
            request_type = parts[0].split('{')[1].split('}')[0].split('=')[1].strip('"')
            time_sum = float(parts[1])
            processing_times[request_type] = time_sum

    return {
        'data': [go.Bar(x=list(processing_times.keys()), y=list(processing_times.values()))],
        'layout': go.Layout(title='Total Processing Time by Request Type')
    }

@app.callback(Output('game-state-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_game_state_graph(n):
    response = requests.get('http://localhost:8000/metrics')
    metrics = response.text.split('\n')
    
    game_states = {}
    for metric in metrics:
        if metric.startswith('game_automation_game_state'):
            parts = metric.split()
            state = parts[0].split('{')[1].split('}')[0].split('=')[1].strip('"')
            value = float(parts[1])
            game_states[state] = value

    return {
        'data': [go.Pie(labels=list(game_states.keys()), values=list(game_states.values()))],
        'layout': go.Layout(title='Current Game State')
    }

if __name__ == '__main__':
    app.run_server(debug=True)