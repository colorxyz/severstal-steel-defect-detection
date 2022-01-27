import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input
from dash_extensions import BeforeAfter  # pip install dash-extensions==0.0.47 or higher
from urllib.parse import quote as urlquote

from dash_extensions import Lottie       # pip install dash-extensions
import dash_bootstrap_components as dbc  # pip install dash-bootstrap-components
import plotly.express as px              # pip install plotly
import pandas as pd                      # pip install pandas
import numpy as np
from matplotlib import pyplot as plt
import mpld3                                # pip install mpld3
import os
import dash                              # pip install dash

import plotly.express as px              # pip install plotly
import matplotlib.patches as patches
import re
import random
import pickle
import cv2
import seaborn as sns
from PIL import Image
import warnings
from datetime import date,datetime
import calendar
warnings.filterwarnings("ignore")

def random_datetimes_or_dates(start, end, out_format='datetime', n=10):

    '''
    unix timestamp is in ns by default.
    I divide the unix time value by 10**9 to make it seconds (or 24*60*60*10**9 to make it days).
    The corresponding unit variable is passed to the pd.to_datetime function.
    Values for the (divide_by, unit) pair to select is defined by the out_format parameter.
    for 1 -> out_format='datetime'
    for 2 -> out_format=anything else
    '''
    (divide_by, unit) = (10**9, 's') if out_format=='datetime' else (24*60*60*10**9, 'D')

    start_u = start.value//divide_by
    end_u = end.value//divide_by
    np.random.seed(0)
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit)


def get_data(arg_data):
    df_csv = pd.read_csv('./input/{}.csv'.format(arg_data))
    df_csv = df_csv.dropna()
    df_folder_path = './input/{}_images'.format(arg_data)
    df_csv['ClassId'] = df_csv['ClassId'].astype(int)

    Image_id = []
    label = []

    for i in os.listdir(df_folder_path):  # https://www.geeksforgeeks.org/python-os-listdir-method/
        for j in range(1, 5):
            Image_id.append(i)
            label.append(j)

    x = {'ImageId': Image_id, 'ClassId': label}  # https://www.geeksforgeeks.org/creating-a-pandas-dataframe/
    df_img = pd.DataFrame(x)

    df = pd.merge(df_img, df_csv, how='outer', on=['ImageId', 'ClassId'])
    df.fillna('', inplace=True)

    df = pd.pivot_table(df, values='EncodedPixels', index='ImageId', columns='ClassId', aggfunc=np.sum).astype(str)
    df = df.reset_index()
    df.columns = ['image_id', 'rle_1', 'rle_2', 'rle_3', 'rle_4']

    defect = []
    stratify = []
    for i in range(len(df)):
        if (df['rle_1'][i] != '' or df['rle_2'][i] != '' or df['rle_3'][i] != '' or df['rle_4'][i] != ''):
            defect.append(1)
        else:
            defect.append(0)

        if df['rle_1'][i] != '':
            stratify.append(1)
        elif df['rle_2'][i] != '':
            stratify.append(2)
        elif df['rle_3'][i] != '':
            stratify.append(3)
        elif df['rle_4'][i] != '':
            stratify.append(4)
        else:
            stratify.append(0)

    df['defect'] = defect
    df['stratify'] = stratify

    defect_1, defect_2, defect_3, defect_4 = [], [], [], []
    for i in range(len(df)):
        if df['rle_1'][i] != '':
            defect_1.append(1)
        else:
            defect_1.append(0)
        if df['rle_2'][i] != '':
            defect_2.append(1)
        else:
            defect_2.append(0)
        if df['rle_3'][i] != '':
            defect_3.append(1)
        else:
            defect_3.append(0)
        if df['rle_4'][i] != '':
            defect_4.append(1)
        else:
            defect_4.append(0)
    df['defect_1'] = defect_1
    df['defect_2'] = defect_2
    df['defect_3'] = defect_3
    df['defect_4'] = defect_4
    df['total_defects'] = df['defect_1'] + df['defect_2'] + df['defect_3'] + df['defect_4']
    start = pd.to_datetime('2021-01-01')
    end = pd.to_datetime('2022-01-01')
    dates = random_datetimes_or_dates(start, end, out_format='datetime', n=df.shape[0])
    df = df.set_index(dates)
    df = df.sort_index()
    return df

def func(v,p): #https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct
    a=p*sum(v)/100
    return "{:.2f}%\n({:.1f})".format(p,a)
def get_defect_info(df):
    defect_1,defect_2,defect_3,defect_4,no_defect=0,0,0,0,0

    for i in range(len(df)):
        if df['rle_1'][i] != '':
            defect_1+=1
        if df['rle_2'][i] != '':
            defect_2+=1
        if df['rle_3'][i] != '':
            defect_3+=1
        if df['rle_4'][i] != '':
            defect_4+=1
        if df['defect'][i] == 0:
            no_defect+=1
    labels=['defect_1','defect_2','defect_3','defect_4','no_defect']
    sizes=[defect_1,defect_2,defect_3,defect_4,no_defect]
    return labels, sizes


def rle_to_mask(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == '') | (rle == '-1'):
        return np.zeros((256, 1600), dtype=np.uint8)

    height = 256
    width = 1600
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2] - 1
    lengths = array[1::2]
    try:
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = 1
    except:
        return mask.reshape((height, width), order='F')
    return mask.reshape((height, width), order='F')


# https://www.kaggle.com/titericz/building-and-visualizing-masks
def rle2maskResize(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == ''):
        return np.zeros((128, 800), dtype=np.uint8)

    height = 256
    width = 1600
    mask = np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2] - 1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1

    return mask.reshape((height, width), order='F')[::2, ::2]


def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:, width:], np.zeros((h, width))], axis=1)
    mask2 = np.logical_xor(mask, mask2)
    mask3 = np.concatenate([mask[width:, :], np.zeros((width, w))], axis=0)
    mask3 = np.logical_xor(mask, mask3)
    return np.logical_or(mask2, mask3)


def mask2pad(mask, pad=2):
    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT
    w = mask.shape[1]
    h = mask.shape[0]

    # MASK UP
    for k in range(1, pad, 2):
        temp = np.concatenate([mask[k:, :], np.zeros((k, w))], axis=0)
        mask = np.logical_or(mask, temp)
    # MASK DOWN
    for k in range(1, pad, 2):
        temp = np.concatenate([np.zeros((k, w)), mask[:-k, :]], axis=0)
        mask = np.logical_or(mask, temp)
    # MASK LEFT
    for k in range(1, pad, 2):
        temp = np.concatenate([mask[:, k:], np.zeros((h, k))], axis=1)
        mask = np.logical_or(mask, temp)
    # MASK RIGHT
    for k in range(1, pad, 2):
        temp = np.concatenate([np.zeros((h, k)), mask[:, :-k]], axis=1)
        mask = np.logical_or(mask, temp)

    return mask

def plot_mask2(arg_data,rle_defect, k):
    df_folder_path = './input/{}_images'.format(arg_data)
    x = rle_defect.columns[2]
    #Create figure and axes
    fig, ax = plt.subplots(3,3,figsize=(9,4 ))
    fig.suptitle('Defect_'+str(k)+'_Images', fontsize=20, fontweight='bold')
    extra = '  has defect '+str(k)
    for i in range(3):
        image_id = rle_defect['image_id'][i]
        rle=rle_defect[x][i]
        im=Image.open(df_folder_path+'/'+str(image_id))
        ax[i,0].imshow(im)
        ax[i,0].set_title(image_id)
        mask=rle_to_mask(rle)
        ax[i,1].imshow(mask)
        ax[i,1].set_title("Mask for "+str(image_id))
        img = np.array(im)
        msk = mask2pad(mask,pad=3)
        msk = mask2contour(msk,width=5)
        img[msk==1,0] = 235
        img[msk==1,1] = 235
        ax[i,2].imshow(img)
        ax[i,2].set_title("Mask2 for "+str(image_id) +extra)
    fig.set_facecolor("tan")
    return fig

# Bootstrap themes by Ann: https://hellodash.pythonanywhere.com/theme_explorer
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = 'SteelDefectDection'
app.layout = dbc.Container([
    # dbc.Row([
    #         dbc.Col([
    #             dbc.Card([dbc.CardImg(src='/asset/SteelDefect.png')]),
    #                 ]),
    #         ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.CardLink("Linked In", target="_blank",
                                 href="https://www.linkedin.com/in/changhyuck-lee-8a144247/"
                    )
                ])
            ],className='mb-2'),
            dbc.Card([
                dbc.CardBody([
                    dbc.CardLink("Git Hub", target="_blank",
                                 href="https://github.com/colorxyz/"
                    )
                ])
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                        dcc.DatePickerRange(
                        id='my-date-picker-range',
                        min_date_allowed=date(1995, 8, 5),
                        max_date_allowed=date(2090, 12, 31),
                        initial_visible_month=datetime.now().date(),
                        start_date='2021-01-01',
                        end_date='2021-12-31',
                        display_format="YYYY-MM-DD",
                    ),
                ])
            ], color="info"),
        ], width=4),
        dbc.Col([
            dcc.Dropdown(
                id='data_dropdown',
                value='train',
                clearable=False,
                options=[{'label': x, 'value': x} for x in ['train','test']])
        ], width=4),
        # dbc.Col([
        #     # html.H1("File Browser"),
        #     # html.H2("Upload"),
        #     dcc.Upload(
        #         id="upload-data",
        #         children=html.Div(
        #             ["Drag and drop or click to select a file to upload."]
        #         ),
        #         style={
        #             "width": "100%",
        #             "height": "60px",
        #             "lineHeight": "60px",
        #             "borderWidth": "1px",
        #             "borderStyle": "dashed",
        #             "borderRadius": "5px",
        #             "textAlign": "center",
        #             "margin": "10px",
        #         },
        #         multiple=True,
        #     ),
        #     html.H2("File List"),
        #     html.Ul(id="file-list"),
        # ], width=4),
    ],className='mb-2 mt-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Defect 1'),
                    html.H2(id='Defect_1', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Defect 2'),
                    html.H2(id='Defect_2', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Defect 3'),
                    html.H2(id='Defect_3', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('Defect 4'),
                    html.H2(id='Defect_4', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6('No Defect'),
                    html.H2(id='Defect_No', children="000")
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(
                    id='defect_dropdown',
                    value='defect_1',
                    clearable=False,
                    options=[{'label': x, 'value': x} for x in ['defect_1','defect_2','defect_3','defect_4']])
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # html.Iframe(
                    # id='pie_chart_Defect_Distribution',
                    # srcDoc=None,  # here is where we will put the graph we make
                    # style={'border-width': '5', 'width': '100%',
                    #        'height': '500px'}),
                    dcc.Graph(id='pie_chart_Defect_Distribution', figure={}, config={'displayModeBar': False}),
                ])
            ]),
        ], width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Iframe(
                    id='plot_image1',
                    srcDoc=None,  # here is where we will put the graph we make
                    style={'border-width': '5', 'width': '100%',
                           'height': '500px'}),
                ])
            ]),
        ], width=7),
    ],className='mb-2'),
], fluid=True)


@app.callback(
    Output('Defect_1', 'children'),
    Output('Defect_2', 'children'),
    Output('Defect_3', 'children'),
    Output('Defect_4', 'children'),
    Output('Defect_No', 'children'),

    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('data_dropdown', 'value'), ]
)
def update_small_cards(start_date, end_date, data):
    df = get_data(data)

    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df[mask].sort_index()
    labels, sizes = get_defect_info(df)
    return sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]


# Pie Chart ************************************************************
@app.callback(
    # Output('pie_chart_Defect_Distribution','srcDoc'),
    Output('pie_chart_Defect_Distribution', 'figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('data_dropdown', 'value'), ]
)
def update_pie(start_date, end_date, data):
    df = get_data(data)

    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df[mask].sort_index()
    labels, sizes = get_defect_info(df)

    fig_pie = px.pie(names=labels, values=sizes,
                     template='ggplot2', title="Defect Distribution"
                     )
    fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig_pie.update_traces(marker_colors=['red', 'blue'])

    return fig_pie


# Plot Image 1************************************************************
@app.callback(
    Output('plot_image1', 'srcDoc'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date'),
     Input('data_dropdown', 'value'),
     Input('defect_dropdown', 'value'), ]
)
def plot_image1(start_date, end_date, data, defect):
    df = get_data(data)

    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df[mask].sort_index()
    rle = 'rle_' + defect[-1]
    rle_defect = df[df[defect] == 1]
    rle_defect = rle_defect[['image_id', rle]]
    rle_defect = rle_defect.sample(n=3)
    rle_defect = rle_defect.reset_index()
    fig = plot_mask2(data, rle_defect, int(defect[-1]))
    html_img1 = mpld3.fig_to_html(fig)
    return html_img1

def set_submission():
    df=pd.read_csv('input/submission.csv')
    df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.split('_', 1, expand=True)
    df = df.drop(columns=['ImageId_ClassId'])
    df.to_csv('input/test.csv')

UPLOAD_DIRECTORY='./input/'
def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        # set_submission()
        return [html.Li(file_download_link(filename)) for filename in files]

if __name__=='__main__':
    app.run_server(debug=False, port=8005)