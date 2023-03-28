
#notes: 
#Packages to install: micropip, shiny, shinylive
#To build page: shinylive export myapp docs
#

#import libraries
import micropip
import asyncio

async def mypackages():
    await micropip.install("scikit-learn")

mypackages()

from shiny import App, render, ui
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import seaborn as sb
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def getFile(dataset):
    if dataset=="hillenbrand":
        infile = Path(__file__).parent / "hillenbrand_normalized.csv"
    else:
        infile = Path(__file__).parent / "buckeye_normalized.csv"
    df = pd.read_csv(infile)
    return df

def plotEllipse(x,y,col="black"):
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  color=col,
                  width=lambda_[0]*2*2, height=lambda_[1]*2*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor('none')
    return ell

def getKmeansClusters(df, nclust):
    kmeans = KMeans(init="k-means++",n_clusters = nclust).fit(df)
    return kmeans.labels_

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
                ui.h2("A boring scatterplot"),
                ui.input_select("algorithm", "Algorithm", {"kmeans": "K-means", "optics": "OPTICS"}),
                ui.input_select("dataset", "Dataset", {"hillenbrand": "Hillenbrand monophthongs", "buckeye": "Buckeye Corpus corner vowels"}),
                ui.input_slider("n_clusters", "Number of clusters (k)", 1, 15, 6)
        ),
        
        # ui.panel_main(
        #         ui.output_plot("plot"),
        #         ui.output_plot("plotVow")

        # )
        ui.panel_main(
            #ui.h3("Hand-labeled data"),
            #ui.output_text("txt"),
            ui.output_plot("plot", width="500px", height="450px"),
            ui.output_plot("plotVow", width="500px", height="450px")
            #ui.img(src="coords.png", style="width: 100%; max-width: 250px;"),
        )
    )
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        n = input.n_clusters()
        val = n*5
        return f'Results: "{val}"'

    @output
    @render.plot()
    def plot():
        palette = sb.color_palette(None, 6)

        df = getFile(input.dataset())

        cats = df['label'].unique()
        ncats = len(cats)
        palette = sb.color_palette(None, ncats)

        fig, ax = plt.subplots()
        for index,value in enumerate(cats):
            points = df[df['label'] == value][['f2','f1']]
            points = points.to_numpy()
            ax.scatter(points[:, 0], points[:, 1], s=50,alpha=0.4, color=palette[index])
            ell = plotEllipse(points[:,0],points[:,1],col=palette[index])
            ax.add_artist(ell)
            #hull = ConvexHull(points)
            #vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
            #ax.plot(points[vert, 0], points[vert, 1], c=palette[index])
            #ax.fill(points[vert, 0], points[vert, 1], c=palette[index], alpha=0.2)
        # ax.scatter(df["f2"], df["f1"], color = color, alpha= alpha)

        # ax.set_title(label = "My title")
        # ax.set_xlabel("X axis label")
        # ax.set_ylabel("Y axis label")
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xlabel('F2 (normalized)')
        plt.ylabel('F1 (normalized)') 
        plt.title('Hand-labeled vowel categories')
        return fig

    @output
    @render.plot()
    def plotVow():
        df = getFile(input.dataset())
        n_clusters = input.n_clusters()
        clust_results = getKmeansClusters(df[['f1','f2']],n_clusters)

        df['k_pred'] = clust_results
        palette = sb.color_palette(None, n_clusters)

        fig, ax = plt.subplots()
        for i in range(n_clusters):
            points = df[df['k_pred'] == i][['f2','f1']]
            points = points.to_numpy()
            ax.scatter(points[:, 0], points[:, 1], s=50,alpha=0.4, color=palette[i], label=f'Cluster {i + 1}')
            hull = ConvexHull(points)
            vert = np.append(hull.vertices, hull.vertices[0])  # close the polygon by appending the first point at the end
            ax.plot(points[vert, 0], points[vert, 1], c=palette[i])
            ax.fill(points[vert, 0], points[vert, 1], c=palette[i], alpha=0.2)

        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xlabel('F2 (normalized)')
        plt.ylabel('F1 (normalized)') 
        plt.title('K-means results')
        return fig

app = App(app_ui, server)


