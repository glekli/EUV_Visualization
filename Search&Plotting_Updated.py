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
header_style = {'font-size': '36px', 'text-align': 'center', 'margin':'0px'}
search_button={'margin-left':'40%', 'height':'50px', 'width':'100px', 'padding': '5px', 'text-align': 'center', 'border-radius': '8px'}

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

        # print("For TESTING:")
        # print("Elements:", element_options)
        # print("beam energies:", beam_energy_options)
        # print("ELEMENTS NAMING IN THE FILE****:", element)
        # print("BEAM ENERGY NAMING IN THE FILE****:", beam_energy)

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

        for group in testf:
            for key in testf[group]:
                ds = testf[group][key]

                element_ds = ds.attrs.get('Element')
                beam_energy_ds = ds.attrs.get('Beam E eV')
                date_str_ds = ds.attrs.get('rawdate')
                time_str_ds = ds.attrs.get('rawtime')
                filename_ds = ds.attrs.get('File name', '')

                # Exclude files without names
                if not filename_ds:
                    continue

                if (
                    (not elements or element_ds in elements) and
                    (not beam_energies or beam_energy_ds in beam_energies)
                ):
                    if (
                        (start_date and date_str_ds and pd.to_datetime(date_str_ds, format='%d%b%Y').date() < pd.to_datetime(start_date).date()) or
                        (end_date and date_str_ds and pd.to_datetime(date_str_ds, format='%d%b%Y').date() > pd.to_datetime(end_date).date())
                    ):
                        continue

                    if not elements or element_ds in elements:
                        if not beam_energies or beam_energy_ds in beam_energies:
                            if pd.notna(element_ds) and pd.notna(beam_energy_ds):  # Check for NaN values
                                spectra_divs.append(generate_spectra_checklists(group, key))

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
def generate_spectra_checklists(group: str, spectrum_key: str):
    return html.Div([
        dcc.Checklist(
            options=[{'label': '', 'value': group + " " + spectrum_key}],
            id='spectrum1-checklist',
            value=[]
        ),
        dcc.Checklist(
            options=[{'label': spectrum_key, 'value': group + " " + spectrum_key}],
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

        for spectrum1 in selected_spectra1:
            group, key = spectrum1.split(' ')
            try:
                ds1 = testf[group][key]
                intensity1 = np.clip(ds1[...][0, :], 0, 7000).tolist()

                scatter_plots1.append(
                    go.Scatter(
                        x=list(range(len(C.apply(ds1[...])))),
                        y=C.apply(ds1[...]),
                        mode='lines',
                        name=ds1.attrs['File name'][:40]
                    )
                )
            except KeyError:
                pass

        if scatter_plots1:
            fig1 = go.Figure(data=scatter_plots1)
            plot1 = dcc.Graph(figure=fig1, id='plot1')
        else:
            plot1 = html.Div("No data to plot for Spectrum 1.")

        for spectrum2 in selected_spectra2:
            group, key = spectrum2.split(' ')
            try:
                ds2 = testf[group][key]
                intensity2 = np.clip(ds2[...][0, :], 0, 7000).tolist()
                scatter_plots2.append(
                    go.Scatter(
                        x=list(range(len(C.apply(ds2[...])))),
                        y=C.apply(ds2[...]),
                        mode='lines',
                        name=ds2.attrs['File name'][:40]
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

    return '', html.Div(), html.Div()

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
