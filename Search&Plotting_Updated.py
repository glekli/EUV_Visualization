import h5py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from math import sqrt
from datetime import datetime as dt
import warnings
import math
import pandas as pd
import time
import json
import requests
import re

testf = h5py.File(r"C:\PHYS\SpecCal\euv_fitting\euv_fitting\euvh5\EUV.h5", 'r')

app = dash.Dash(__name__)

# Set suppress_callback_exceptions=True
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Store(id='Graph_Info', storage_type='local'),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Link(
        rel="stylesheet",
        href="/assets/style.css"
    ),
])

# Callback to update the page content based on the pathname
@app.callback(
        Output('page-content', 'children'),
        [Input('url', 'pathname')]
        )
def display_page(pathname):
    if pathname == '/search':
        return search_layout
    elif pathname == '/plotting':
        return plotting_layout
    else:
        return home_layout

# Define CSS styles for elements
element_label_style = {'font-weight': 'bold'}
dropdown_style = {'width': '80%', 'padding': '5px', 'box-sizing': 'border-box'}
input_style = {'width': '50%'}
button_style = {'className': 'btn btn-primary'}
header_style = {'fontSize': '36px', 'textAlign': 'center', 'margin':'0px'}
search_button = {'margin-left':'40%', 'height':'50px', 'width':'100px', 'padding': '5px', 'textAlign': 'center', 'border-radius': '8px'}

# Define the layout for the home page
home_layout = html.Div([
    html.H1('Home Page', style=header_style),
    html.Div([
        html.A('Search', href='/search', **button_style),
        html.Br(),
        html.Br(),
        html.A('Plotting', href='/plotting', **button_style),
        html.Br(),
    ])
])

# Define the layout for the search page
search_layout = html.Div(className='body', children=[
    html.H1('Search Page', style=header_style),
    html.Div(className='search-box', children=[
        html.Div([
            html.Label('Element:' ,style=element_label_style),
            html.Div(className='element', children =[
            dcc.Dropdown(
                id='element-input',
                multi=True,
                placeholder='Select Chemical Symbols',
                style=dropdown_style
            ),
            ]),
            html.Br(),
            html.Label('Beam Energy:', style=element_label_style),
            html.Div(className='beam_energy', children = [
            dcc.Dropdown(
                id='beam-energy-input',
                multi=True,
                placeholder='Select Beam Energies',
                style=dropdown_style
            )
            ]),
        ]),
# style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px'}
        html.Div(id='start-date', children=[
            html.Label("Select Start Date:", style=element_label_style),
            dcc.DatePickerSingle(
                id='start-date-picker',
                display_format='YYYY-MM-DD',
                date=dt(2020, 1, 6),
            ),
            html.Label("Time:", style=element_label_style),
            dcc.Input(
                id='start-time-picker',
                type='text',
                placeholder='Enter time in format HH:MM:SS',
                value='00:00:00'
            )
        ]),
        html.Br(),
        html.Div(id='end-date', children=[
            html.Label("Select End Date:", style=element_label_style),
            dcc.DatePickerSingle(
                id='end-date-picker',
                display_format='YYYY-MM-DD',
                date=dt(2023, 1, 7),
            ),
            html.Label("Time:", style=element_label_style),
            dcc.Input(
                id='end-time-picker',
                type='text',
                placeholder='Enter time in format HH:MM:SS',
                value='00:00:00'
            )
        ]),
        html.Br(),
        html.Button('Search', id='search-button', n_clicks=0, **button_style, style=search_button)
    ]),
    html.Div(id='results-container'),
    html.Div(id='selected-spectra-container'),
    html.Br(),
    html.A('Back to Home', href='/')
])
# Define the layout for the plotting page
plotting_layout = html.Div([
    html.H1('Plotting Page', style=header_style),
    dcc.Checklist(
            options=[],
            id='spectra-checklist',
            value=[]
    ),
        html.Div(id='plot-container1'),
        html.Div(id='plot-container2'),
        html.A('Back to Home', href='/', **button_style)

])

@app.callback(
    [Output('element-input', 'options'),
     Output('beam-energy-input', 'options')],
    [Input('url', 'pathname'), Input('element-input', 'value')],
)
def populate_dropdown_options(pathname, selected_elements):
    if pathname == '/search':
        elements = set()
        beam_energies_dict = {} #maps beam energies to elements

        for group in testf:
            for key in testf[group]:
                ds = testf[group][key]

                element = ds.attrs.get('Element')
                beam_energy = ds.attrs.get('Beam E eV')

                if isinstance(element, str):
                    element = element.strip()

                elif isinstance(element, (np.float64, float, int)):
                    element = str(element).strip()

                if not element:
                    continue

                elements.add(element)

                # Check if beam_energy is a valid number
                if selected_elements and (element not in selected_elements):
                    continue

                if isinstance(beam_energy, (np.float64, float, int)) and not np.isnan(beam_energy):
                    beam_energy = int(beam_energy)

                    # Save each unique element for each beam energy
                    if beam_energy not in beam_energies_dict:
                        beam_energies_dict[beam_energy] = set()

                    beam_energies_dict[beam_energy].add(element)


        # Create a list of dictionaries for element options
        element_options = [{'label': elem, 'value': elem} for elem in sorted(list(elements))]

        # Create a list of dictionaries for beam energy options in ascending order
        beam_energy_options = [{'label': f"{energy} - '{', '.join(beam_energies_dict[energy])}'", 'value': energy} for energy in sorted(beam_energies_dict.keys())]

        return element_options, beam_energy_options

    return [], []


@app.callback(
    Output('results-container', 'children'),
    [Input('search-button', 'n_clicks')],
    [
        State('element-input', 'value'),
        State('beam-energy-input', 'value'),
        State('start-date-picker', 'date'),
        State('start-time-picker', 'value'),
        State('end-date-picker', 'date'),
        State('end-time-picker', 'value')
    ]
)
def update_search_results(n_clicks, elements, beam_energies, start_date, start_time, end_date, end_time):
    if n_clicks > 0:
        spectra_divs = []

        query = {}
        if elements:
            query['element'] = elements
        if beam_energies:
            query['beam_energy'] = beam_energies
        if start_date:
            start_match = re.search('([0-9]+)-([0-9]+)-([0-9]+)', start_date)
            if start_match is not None:
                query['lower_date_taken'] = start_match.group(1) + '/' + start_match.group(2) + '/' + start_match.group(3)
        if end_date:
            end_match = re.search('([0-9]+)-([0-9]+)-([0-9]+)', end_date)
            if end_match is not None:
                query['upper_date_taken'] = end_match.group(1) + '/' + end_match.group(2) + '/' + end_match.group(3)

        result_raw = requests.get(f'http://127.0.0.1:8000/spectra/metadata?page=0&per_page=0', params=query)
        result = json.loads(json.loads(result_raw.content))

        if result['records']:
            for file_name, spectra_id in zip(result['records']['file_name'], result['records']['spectra_id']):
                spectra_divs.append(generate_spectra_checklists(spectra_id, file_name))

        if spectra_divs:
            return [
                html.H4('Plot'),
                *spectra_divs,
                html.Button('Plot Selected Spectra', id='plot-button', n_clicks=0),
                html.Div(id='redirect-container')
            ]
        else:
            return html.P('No matching spectra found.')

    return ''

# Generate dynamic checklist divs
def generate_spectra_checklists(spectra_id: str, spectrum_key: str):
    return html.Div([
        dcc.Checklist(
            options=[{'label': '', 'value': spectra_id}],
            id='spectrum1-checklist',
            value=[]
        ),
        dcc.Checklist(
            options=[{'label': spectrum_key, 'value': spectra_id}],
            id='spectrum2-checklist',
            value=[]
        ),
    ], style={'display': 'flex', 'align-items': 'center', 'margin': '15px'})

# Callback to handle selected spectra for plotting and redirect to the plotting page
@app.callback(
    [Output('redirect-container', 'children'),
     Output('Graph_Info', 'data')],
    [Input('plot-button', 'n_clicks')],
    [State('results-container', 'children')]
)
def redirect_and_update_data(n_clicks, results_children):
    spectrum1_values, spectrum2_values = [], []
    for child in results_children[1:-2]:
        checklists = child['props']['children']
        spectrum1_values += checklists[0]['props']['value']
        spectrum2_values += checklists[1]['props']['value']

    selected_spectra1 = spectrum1_values
    selected_spectra2 = spectrum2_values

    if n_clicks > 0 and (selected_spectra1 or selected_spectra2):
        return dcc.Location(pathname='/plotting', id='redirect', refresh=True), {
            'selected_spectra1': selected_spectra1,
            'selected_spectra2': selected_spectra2
        }

    return '', {'selected_spectra1': []}

@app.callback(
    [Output('plot-container1', 'children'),
     Output('plot-container2', 'children')],
    [Input('url', 'pathname')],
    [State('Graph_Info', 'data')]
)
def update_plot(pathname, data):
    C = CosmicRayFilter()
    if pathname == '/plotting' and data and 'selected_spectra1' in data:
        selected_spectra1 = data['selected_spectra1']
        selected_spectra2 = data['selected_spectra2']
        scatter_plots1 = []
        scatter_plots2 = []

        if not selected_spectra1 and not selected_spectra2:
            return 'Select spectra to plot', [], []

        query = {}
        query['ids'] = ','.join(str(value) for value in (selected_spectra1))
        result1_raw = requests.get(f'http://127.0.0.1:8000/spectra/data?page=0&per_page=0', params=query)
        result1 = json.loads(json.loads(result1_raw.content))

        query['ids'] = ','.join(str(value) for value in (selected_spectra2))
        result2_raw = requests.get(f'http://127.0.0.1:8000/spectra/data?page=0&per_page=0', params=query)
        result2 = json.loads(json.loads(result2_raw.content))

        for dataset_key in result1['records'].keys():
            plot_data = np.array(result1['records'][dataset_key]['data'])
            plot_data = np.reshape(plot_data, (1, 2048))
            try:
                scatter_plots1.append(
                    go.Scatter(
                        x=list(range(len(C.apply(plot_data[...])))),
                        y=C.apply(plot_data[...]),
                        mode='lines',
                        name=result1['records'][dataset_key]['file_name'][:40]
                    )
                )
            except KeyError:
                pass

        if scatter_plots1:
            fig1 = go.Figure(data=scatter_plots1)
            plot1 = dcc.Graph(figure=fig1, id='plot1')
        else:
            plot1 = html.Div("No data to plot for Spectrum 1.")

        for dataset_key in result2['records'].keys():
            plot_data = np.array(result2['records'][dataset_key]['data'])
            plot_data = np.reshape(plot_data, (1, 2048))
            try:
                scatter_plots2.append(
                    go.Scatter(
                        x=list(range(len(C.apply(plot_data[...])))),
                        y=C.apply(plot_data[...]),
                        mode='lines',
                        name=result2['records'][dataset_key]['file_name'][:40]
                    )
                )
            except KeyError:
                pass

        if scatter_plots2:
            fig2 = go.Figure(data=scatter_plots2)
            plot2 = dcc.Graph(figure=fig2, id='plot2')
        else:
            plot2 = html.Div("No data to plot for Spectrum 2.")

        return plot1, plot2

    return html.Div(), html.Div()

class CosmicRayFilter:
    """
    Cosmic Ray Filter class for removal of cosmic rays from spectra.
    """
    def __init__(self, filterval=5):
        self.filterval = filterval

    def apply(self, data, combine=True):
        """
        Apply filter to input data.

        Returns the spectra with cosmic rays removed.
        """
        data = data.astype(float)
        nr_frames, camera_size = data.shape

        if nr_frames == 1:
            warnings.warn("Only 1 frame, can't detect cosmic rays", RuntimeWarning)
            return data[0]

        for pixel in range(camera_size):
            allframes = data[:, pixel]
            cosmic_frames = []
            for frame in range(nr_frames):
                testval = allframes[frame]
                testframes = np.delete(allframes, frame)
                poisson_noise = sum([sqrt(val) for val in testframes]) / (nr_frames - 1)
                mean = sum(testframes) / (nr_frames - 1)

                if testval > (mean + (self.filterval * poisson_noise)):
                    cosmic_frames.append(frame)

            if len(cosmic_frames) > 0:
                non_cosmic_avg = np.mean(np.delete(allframes, cosmic_frames))
                data[cosmic_frames, pixel] = non_cosmic_avg

        if combine:
            return np.sum(data, axis=0)
        else:
            return data

if __name__ == '__main__':
    app.run_server(debug=True)
