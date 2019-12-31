import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

input_file = '../processed_data/prophet_res.pkl'
date_col = 'ds'
y_col = 'y'
store_col = 'store'
item_col = 'item'

df = pd.read_pickle(input_file)
df.sort_values(date_col, inplace=True)

stores = np.sort(df[store_col].unique())
items = np.sort(df[item_col].unique())


# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            className="app-header",
            children=[
                html.Div('Demand data visulalization', className="app-header--title")
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.P('Select store'),
                                dcc.Dropdown(
                                    id='store-id',
                                    options=[{'label': i, 'value': i} for i in stores],
                                    value=stores[0]
                                    )
                            ],
                            className='three columns'
                        ),
                        html.Div(
                            [
                                html.P('Select item'),
                                dcc.Dropdown(
                                    id='item-id',
                                    options=[{'label': i, 'value': i} for i in items],
                                    value=items[0]
                                    )
                            ],
                            className='three columns'
                        )
                    ],
                    className='seven columns'
                ),
                html.Div(
                    [
                        dcc.Graph(id='y_time_series', className='ts_plot')
                        # dcc.Graph(id='y_prediction', className='ts_plot'),
                        # dcc.Graph(id='trend', className='ts_plot'),
                        # dcc.Graph(id='weekly', className='ts_plot'),
                        # dcc.Graph(id='yearly', className='ts_plot')
                    ],
                    className='six columns'
                            # style={'width': '600px'}),
                )
            ]
        )
    ]
)



@app.callback(
    Output('item-id', 'options'),
    [Input('store-id', 'value')])
def set_item_options(selected_store):
    return [{'label': i, 'value': i} for i in np.sort(df.loc[df[store_col] == selected_store, item_col].unique())]


@app.callback(
    Output('item-id', 'value'),
    [Input('store-id', 'options')])
def set_item_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output('y_time_series', 'figure'),
     # Output('x_time_series', 'figure'),
     # Output('x_y_scatter', 'figure')],
    [Input('store-id', 'value'),
     Input('item-id', 'value')])
def update_figure(store_id, item_id):
    filtered_df = df[(df[store_col] == store_id) & (df[item_col] == item_id)]

    return {
        'data': [dict(
            x=filtered_df[date_col],
            y=filtered_df[y_col],
            mode='markers',
            marker=dict(color='blue', size=5, opacity=0.5)
        )],
        'layout': dict(
            xaxis={
                'title': date_col,
            },
            yaxis={
                'title': y_col,
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
