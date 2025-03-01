import dash
from dash import dcc, html, Dash, Input, Output, State, no_update
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objs as go
import xgboost as xgb

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions=True
server = app.server

df = pd.read_csv("data.csv")
label_enc = LabelEncoder()
original = df['label'].unique()
original.sort()
df['label'] = label_enc.fit_transform(df['label'])
label_map = {enc : ori for enc, ori in enumerate(original)}

X = df.drop(columns=['label'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', C=10, probability=True, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

model_performance = {}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)


    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)

    fpr = {}
    tpr = {}
    roc_auc = {}
    n_classes = len(np.unique(y_test))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            (y_test == i).astype(int), 
            y_pred_proba[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])

    model_performance[model_name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred_test),
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }



def describe_clusters(cluster_centers, clusters):

    overall_means = cluster_centers.mean()
    overall_std = cluster_centers.std()

    clusters_desc = {}
    clusters_crops = {}
    
    
    for idx, center in enumerate(cluster_centers):
        features = []

        # If a feature is greater than mean by 30% then consider High, else if less than mean by 30% consider Low
        # Else consider Moderate

        npk_values = center[:3]
        npk_mean = npk_values.mean()
        
        if npk_mean > df[['N', 'P', 'K']].mean().mean() * 1.3:
            features.append("Nutrient-Rich")
        elif npk_mean < df[['N', 'P', 'K']].mean().mean() * 0.7:
            features.append("Nutrient-Poor")
        else:
            features.append("Balanced-Nutrients")
        
        
        if center[3] > center[3].mean() * 1.3:
            features.append("Warm-Climate")
        elif center[3] < center[3] * 0.7:
            features.append("Cool-Climate")
        else:
            features.append("Moderate-Climate")
        
        if center[4] > center[4].mean() * 1.3:
            features.append("High-Humidity")
        elif center[4] < center[4] * 0.7:
            features.append("Low-Humidity")
        else:
            features.append("Moderate-Humidity")
        
        if center[5] > 7.5:
            features.append("Alkaline-Soil")
        elif center[5] < 5.5:
            features.append("Acidic-Soil")
        else:
            features.append("Neutral-Soil")
        
        if center[6] > center[6].mean() * 1.3:
            features.append("High-Rainfall")
        elif center[6] < center[6] * 0.7:
            features.append("Low-Rainfall")
        else:
            features.append("Moderate-Rainfall")
        
        clusters_desc[idx] = '\t'.join(features)
        clusters_crops[idx] = list(df[clusters == idx]['label'].map(label_map).unique())
    return clusters_desc, clusters_crops



kmeans_info = {}
intertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    silhouette_avg = silhouette_score(X, clusters)
    intertias.append(kmeans.inertia_)
    desc, crops = describe_clusters(centers, clusters)
    kmeans_info[k] = {
        'silhouette_score':silhouette_avg,
        'clusters': clusters,
        'clusters_desc': desc,
        'clusters_crops': crops
    }

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train)



app.layout = html.Div([
    html.H1("Agricultural Production Optimization", style={'textAlign':'center'}),
    
    html.Div([
        html.H3("Data Overview"),
        dcc.Graph(id='feature-relationships'),
        dcc.Dropdown(
            id='feature-pair-dropdown',
            options=[{'label': f'{col1} vs {col2}', 
                     'value': f'{col1},{col2}'} for i, col1 in enumerate(X.columns) 
                                               for col2 in X.columns[i+1:]],
            value=f'{X.columns[0]},{X.columns[1]}'
        ),
        
        html.Div([
            html.Div([
                dcc.Graph(id='distribution-plots'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in X.columns],
                    value=X.columns[0]
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='correlation-matrix')
            ], style={'width': '48%', 'display': 'inline-block'})
        ]),
        
    ]),
    
    html.Hr(),
    
    html.Div([
        html.H3("Classification"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': k, 'value': k} for k in models.keys()],
            value='Random Forest'
        ),
        html.Button('Show Model Performance', id='show-performance-button', n_clicks=0),
        html.Div(id='performance-metrics'),
        html.Div(id='train-test'),
        html.Div(
            [dcc.Graph(id='confusion-matrix'),
        dcc.Graph(id='roc-curve')],
        style={
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'space-between',
        })
    ]),
    
    html.Hr(),
    
    html.Div([
      html.H3("Clustering"),
      dcc.Graph(id='elbow-plot'),
      html.Button('Find Optimal K', id='elbow-button', n_clicks=0),
      dcc.Graph(id='cluster-plot'),
      html.Button('Run Clustering', id='cluster-button', n_clicks=0),
      dcc.Slider(
          id='k-slider',
          min=2,
          max=10,
          step=1,
          value=3,
          marks={i: str(i) for i in range(2, 11)}
      ),
      html.Div(id='clusters-info'),
      html.Div(id='silhouette-score', children="Silhouette Score: N/A")
  ]),

    html.Hr(),

    html.Div([
        html.H3("Model Comparison"),
        html.Table(id='model-comparison-table')
    ])
])

@app.callback(
    Output('feature-relationships', 'figure'),
    [Input('feature-pair-dropdown', 'value')]
)
def update_feature_relationship(feature_pair):
    col1, col2 = feature_pair.split(',')
    fig = px.scatter(df, x=col1, y=col2, color=df['label'].map(label_map),
                    title=f'Feature Relationship: {col1} vs {col2}')
    return fig

@app.callback(
    Output('distribution-plots', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_distribution(feature):
    fig = go.Figure()
    for label in sorted(y.unique()):
        fig.add_trace(go.Histogram(
            x=X[y == label][feature],
            name=label_map[label],
            opacity=0.7
        ))
    fig.update_layout(
        title=f'Distribution of {feature} by Class',
        barmode='overlay'
    )
    return fig

@app.callback(
    Output('correlation-matrix', 'figure'),
    [Input("feature-dropdown", "value")]
)
def update_correlation_matrix(input_value):
    corr = df.corr()
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        title="Feature Correlation Matrix",
        text_auto=True
    )
    fig.update_layout(width=550, height=550, margin=dict(l=50, r=50, t=50, b=50))
    return fig

@app.callback(
    [Output('performance-metrics', 'children'),
    Output('train-test', 'children'),
    Output('confusion-matrix', 'figure')],
    Output('roc-curve', 'figure'),
    [Input('show-performance-button', 'n_clicks')],
    [State('model-dropdown', 'value')]
)
def show_model_performance(n_clicks, model_name):
    if n_clicks == 0:
        return "Select a model and click 'Show Model Performance' to see results.", {}, {}, {}
    
    report = model_performance[model_name]['classification_report']
    cm = model_performance[model_name]['confusion_matrix']
    
    cm_fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix",
                       text_auto=True
                       )
    cm_fig.update_layout(width=700, height=700)


    roc_fig = go.Figure()
    n_classes = len(fpr)
    
    colors = ['#3182bd', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', 
            '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', 
            '#969696', '#bdbdbd', '#d9d9d9', '#d9d9d9']
    for i in range(n_classes):
        roc_fig.add_trace(
            go.Scatter(
                x=model_performance[model_name]['fpr'][i], 
                y=model_performance[model_name]['tpr'][i],
                name=f"{label_map[i]} (AUC = {model_performance[model_name]['roc_auc'][i]:.4f})",
                line=dict(color=colors[i], width=2)
            )
        )
    
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            name='Random Guess',
            line=dict(color="gray", width=2, dash="dash")
        )
    )
    
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=800,
        
    )

    train_acc = model_performance[model_name]['train_accuracy']
    test_acc = model_performance[model_name]['test_accuracy']

    return html.Pre(report), html.Div([html.P(f"Training Accuracy: {train_acc:.2f}"),html.P(f"Testing Accuracy: {test_acc:.2f}")]), cm_fig, roc_fig

@app.callback(
    Output('elbow-plot', 'figure'),
    [Input('elbow-button', 'n_clicks')]
)
def plot_elbow_method(n_clicks):
    if n_clicks == 0:
        return {}
    
    fig = go.Figure(data=go.Scatter(x=list(range(1, 11)), y=intertias, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method for Optimal k',
        xaxis_title='k',
        yaxis_title='Inertia'
    )

    return fig

@app.callback(
    [Output('cluster-plot', 'figure'),
     Output('silhouette-score', 'children'),
     Output('clusters-info', 'children')],
    [Input('cluster-button', 'n_clicks')],
    [State('k-slider', 'value')]
)
def update_clustering(n_clicks, k):
    if n_clicks == 0:
        return {}, "Silhouette Score: N/A", "Clusters Descriptions: N/A"

    fig = go.Figure(data=[go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=kmeans_info[k]['clusters'],
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        )
    )])
    
    fig.update_layout(
        title=f'KMeans Clustering (k={k}) - PCA 3D Visualization',
        scene=dict(
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
            zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    clusters_info = []
    for i, desc in kmeans_info[k]['clusters_desc'].items():
        clusters_info.append(
            html.Div([
                html.P(f"Cluster {i+1}: {desc}"),
                html.P(f"Crops: {', '.join(kmeans_info[k]['clusters_crops'][i])}"),
                html.Br()
            ])
        )
    sill_score = kmeans_info[k]['silhouette_score']
    return fig, f'Silhouette Score: {sill_score:.2f}', clusters_info

@app.callback(
    Output('model-comparison-table', 'children'),
    [Input('show-performance-button', 'n_clicks')]
)
def update_model_comparison(n_clicks):
    if n_clicks == 0:
        return ""
    
    headers = ['Model', 'Training Accuracy', 'Testing Accuracy', 'Precision', 'Recall', 'F1 Score']
    rows = []
    
    for model_name, performance in model_performance.items():
        rows.append([
            model_name,
            str(round(performance['train_accuracy'], 2)),
            str(round(performance['test_accuracy'], 2)),
            str(round(performance['precision'], 2)),
            str(round(performance['recall'], 2)),
            str(round(performance['f1'], 2))
        ])
    
    rows.sort(key=lambda x: float(x[1]), reverse=True)
    
    table_header = [html.Tr([html.Th(col) for col in headers])]
    table_rows = [html.Tr([html.Td(cell) for cell in row]) for row in rows]
    
    return html.Table(table_header + table_rows, style={'width': '100%', 'border': '1px solid black'})

if __name__ == '__main__':
    app.run_server(debug=True)