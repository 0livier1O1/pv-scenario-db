import dash
from dash import Dash, html, dcc, Output, Input, State
import pandas as pd
import plotly.graph_objs as go
import torch

# DUMBO Imports 
from botorch.utils.multi_objective.pareto import is_non_dominated
from utils import drop_null_obj, pv_loads_to_input

feeders = ["p4rhs8"]
feeder = feeders[0]  # Allow for different feeders

# Data 
pv_scens, x = pv_loads_to_input(feeder)
main_ss = feeder + "_69"

buses = pd.read_csv(f"{feeder}/buses.csv", index_col=0)
lines = pd.read_csv(f"{feeder}/lines.csv", index_col=0)

f_b_df0 = pd.read_csv(f"{feeder}/buses_objs.csv", index_col=0)
f_l_df0 = pd.read_csv(f"{feeder}/lines_objs.csv", index_col=0)

f_b0 = torch.tensor(f_b_df0.values, dtype=torch.float64)
f_l0 = torch.tensor(f_l_df0.values, dtype=torch.float64)

y0 = torch.cat([f_b0, f_l0], axis=1)
f_b_n, b_null = drop_null_obj(f_b0)
f_l_n, l_null = drop_null_obj(f_l0)

failed_lines = f_l_df0.columns[~l_null]
failed_buses = f_b_df0.columns[~b_null]

critical_scenarios = list(is_non_dominated(y0).float().nonzero().squeeze().tolist())

# Initialize the app
app = Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Button('All Buses', id='toggle-buses'),
            html.Button('Show Bus Violations', id='show-bus-violations')
        ], style={'display': 'flex', 'flex-direction': 'column', 'margin-right': '1em'}),
        html.Div([
            html.Button('All Lines', id='toggle-lines'),
            html.Button('Show Line Violations', id='show-line-violations')
        ], style={'display': 'flex', 'flex-direction': 'column'})
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'margin-bottom': '2em'}),

    html.Div([
        html.Div([
            dcc.Store(id='buses-state', data={'show_state': 'All Buses', 'show_violations': False}),
            dcc.Store(id='lines-state', data={'show_state': 'All Lines', 'show_violations': False}),
            html.Div([    
                html.Div([
                    dcc.Dropdown(
                        id="feeder-left",
                        options=[{'label': feeder, 'value': feeder} for feeder in feeders],
                        value=feeders[0],
                        style={'width': '100%', 'margin-bottom': '1em'}
                    ),
                    dcc.Dropdown(
                        id='scenario-left',
                        options=[{'label': f'Scenario {i}', 'value': i} for i in critical_scenarios],
                        value=critical_scenarios[0],
                        style={'width': '100%'}
                    )
                ], style={'display': 'flex', 'flex-direction': 'column', 'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                dcc.Graph(id='ntwrk-plot-left', style={'height': '30em', 'margin-bottom': '0', 'padding-bottom': '0'}),
                dcc.Graph(id='histogram-plot-left', style={'height': '20em', 'margin-top': '0', 'padding-top': '0'})
            ], style={'overflow': 'auto'})
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id="feeder-right",
                        options=[{'label': feeder, 'value': feeder} for feeder in feeders],
                        value=feeders[0],
                        style={'width': '100%', 'margin-bottom': '1em'}
                    ),
                    dcc.Dropdown(
                        id='scenario-right',
                        options=[{'label': f'Scenario {i}', 'value': i} for i in critical_scenarios],
                        value=critical_scenarios[0],
                        style={'width': '100%'}
                    )
                ], style={'display': 'flex', 'flex-direction': 'column', 'width': '100%'})
            ], style={'width': '100%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                dcc.Graph(id='ntwrk-plot-right', style={'height': '30em', 'margin-bottom': '0', 'padding-bottom': '0'}),
                dcc.Graph(id='histogram-plot-right', style={'height': '20em', 'margin-top': '0', 'padding-top': '0'})
            ], style={'overflow': 'auto'})
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})

    ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%', 'overflow-x': 'scroll'})
])


def histogram_scen(scen):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=failed_buses, y=f_b_n[scen, :], marker_color="blue", name="Buses", yaxis="y1",
        hoverinfo="text", text=failed_buses)
    )
    fig.add_trace(
        go.Bar(x=failed_lines, y=f_l_n[scen, :], marker_color="red", name="Lines", yaxis="y2",
               hoverinfo="text", text=failed_lines)
    )
    
    all_labels = f_b_df0.columns[~b_null].to_list() + f_l_df0.columns[~l_null].to_list()
    ticktext = [f"<i>{label}</i>" for label in all_labels]
    fig.update_layout(
        width=800,
        height=400, 
        yaxis=dict(
            title='Excess voltage'
        ),
        yaxis2=dict(
            title='Excess flow (%)',
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            tickangle=-45,  # Tilt the labels
            tickmode='array',
            tickvals=list(range(len(all_labels))),
            ticktext=ticktext
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0}
    )

    return fig


def ntwrk_scen(scen, show_buses="All Buses", show_only_violations=False, show_lines="All Lines", show_only_line_violations=False):
    if scen is None:
        scen = critical_scenarios[0]
    scen = int(scen)
    adopters = pv_scens.iloc[scen, :]
    adopters = adopters[adopters == 1].index.get_level_values(1).to_list()

    bus_viols_scen = failed_buses[f_b_n[scen, :] > 0].tolist()
    lines_viols_scen = failed_lines[f_l_n[scen, :] > 0].tolist()

    fig = go.Figure()

    viol_line_traces = []
    line_traces = []
    for _, row in lines.iterrows():
        dict_lines = dict(y=[row['coordinateY'], row['coordinateY_bus2']],
                          x=[row['coordinateX'], row['coordinateX_bus2']], mode="lines", showlegend=False)
        if row["name"] in lines_viols_scen and show_only_line_violations:
            dict_lines.update(dict(line={"color": "red", "width": 5}))
            viol_line_traces.append(go.Scatter(**dict_lines))
        else:
            dict_lines.update(dict(line={"color": "black", "width": 1.5}))
            line_traces.append(go.Scatter(**dict_lines))

    if show_lines == "All Lines":
        fig.add_traces(line_traces)
        if show_only_line_violations:
            fig.add_traces(viol_line_traces)
    else:
        if show_only_line_violations:
            fig.add_traces(viol_line_traces)
    
    if show_buses == "Hide Buses":
        sizes = [0 for bus in buses["name"]]
    elif show_buses == "Adopters Only":
        sizes = [15 if bus in adopters else 0 for bus in buses["name"]]
    else:
        sizes = [15 for bus in buses["name"]]
    color = ["orange" if bus in adopters else "black" for bus in buses["name"]]
    fig.add_trace(go.Scatter(
        y=buses["coordinateY"],
        x=buses["coordinateX"],
        mode="markers",
        hoverinfo="text",
        text=buses["name"],
        marker=dict(size=sizes, color=color),
        showlegend=False,
    ))

    if show_only_violations:
        edges = ["orange" if bus in adopters else "blue" for bus in bus_viols_scen]
        violated_buses = buses[buses["name"].isin(bus_viols_scen)]
        fig.add_trace(go.Scatter(
            y=violated_buses["coordinateY"],
            x=violated_buses["coordinateX"],
            hoverinfo="text",
            text=violated_buses["name"],
            mode="markers", 
            marker=dict(size=20, color="blue", line=dict(color=edges, width=4)),
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
        y=[None],
        x=[None],
        mode="markers", 
        marker=dict(size=20, color="orange"),
        name="Adopters",
        showlegend=True
    ))

    fig.add_trace(go.Scatter(
        y=[None],
        x=[None],
        mode="markers", 
        marker=dict(size=20, color="Blue"),
        name="Violations",
        showlegend=True
    ))
    fig.update_layout(
        showlegend=True,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode="closest",
        dragmode="zoom"
    )

    return fig

@app.callback(
    [Output('histogram-plot-left', 'figure'), Output('ntwrk-plot-left', 'figure')],
    [Input('scenario-left', 'value'), Input('buses-state', 'data'), Input('lines-state', 'data')]
)
def update_scenario_left(selected_scenario, buses_state, lines_state):
    if selected_scenario is None:
        selected_scenario = critical_scenarios[0]
    selected_scenario = int(selected_scenario)

    show_buses = buses_state['show_state']
    show_only_violations = buses_state['show_violations']
    show_lines = lines_state['show_state']
    show_only_lines_violations = lines_state['show_violations']

    histogram_fig = histogram_scen(selected_scenario)
    ntwrk_fig = ntwrk_scen(selected_scenario, show_buses, show_only_violations, show_lines, show_only_lines_violations)

    return histogram_fig, ntwrk_fig


@app.callback(
    [Output('histogram-plot-right', 'figure'), Output('ntwrk-plot-right', 'figure')],
    [Input('scenario-right', 'value'), Input('buses-state', 'data'), Input('lines-state', 'data')]
)
def update_scenario_right(selected_scenario, buses_state, lines_state):
    if selected_scenario is None:
        selected_scenario = critical_scenarios[0]
    selected_scenario = int(selected_scenario)

    show_buses = buses_state['show_state']
    show_only_violations = buses_state['show_violations']
    show_lines = lines_state['show_state']
    show_only_lines_violations = lines_state['show_violations']

    histogram_fig = histogram_scen(selected_scenario)
    ntwrk_fig = ntwrk_scen(selected_scenario, show_buses, show_only_violations, show_lines, show_only_lines_violations)

    return histogram_fig, ntwrk_fig


@app.callback(
    [Output('buses-state', 'data'), Output("toggle-buses", "children"), Output("show-bus-violations", "children")],
    [Input('toggle-buses', 'n_clicks'), Input('show-bus-violations', 'n_clicks')],
    [State('buses-state', 'data')]
)
def update_buses_state(toggle_buses_clicks, show_violations_clicks, state):
    ctx = dash.callback_context
    if state['show_violations']:
        button_label2 = "Hide Bus Violations"
    else:
        button_label2 = "Show Bus Violations"

    if not ctx.triggered:
        return state, state["show_state"], button_label2

    prop_id = ctx.triggered[0]['prop_id']
    button_label1 = state['show_state']

    if 'toggle-buses' in prop_id:
        if state['show_state'] == 'All Buses':
            state['show_state'] = 'Adopters Only'
            button_label1 = 'Adopters Only'
        elif state['show_state'] == 'Adopters Only':
            state['show_state'] = 'Hide Buses'
            button_label1 = 'Hide Buses'
        else:
            state['show_state'] = 'All Buses'
            button_label1 = 'All Buses'
        state['show_violations'] = False

    elif 'show-bus-violations' in prop_id:
        state['show_violations'] = not state['show_violations']

    return state, button_label1, button_label2


@app.callback(
    [Output('lines-state', 'data'), Output('toggle-lines', 'children'), Output('show-line-violations', 'children')],
    [Input('toggle-lines', 'n_clicks'), Input('show-line-violations', 'n_clicks')],
    [State('lines-state', 'data')]
)
def update_lines_state(toggle_lines_clicks, show_line_viol_clicks, state):
    
    if state['show_violations']:
        button_label2 = "Show Line Violations"
    else:
        button_label2 = "Hide Line Violations"

    ctx = dash.callback_context
    if not ctx.triggered:
        return state, state['show_state'], button_label2
    
    prop_id = ctx.triggered[0]['prop_id']

    button_label = state["show_state"]
    if 'toggle-lines' in prop_id:
        if state["show_state"] == "All Lines":
            state["show_state"] = "Hide Lines"  # Corrected here
            button_label = "Hide Lines"
        else:
            state["show_state"] = "All Lines"  # Corrected here
            button_label = "All Lines"
    elif 'show-line-violations' in prop_id:
        state['show_violations'] = not state['show_violations']

    return state, button_label, button_label2 
    

if __name__ == '__main__':
    app.run(debug=True)
