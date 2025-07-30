import matplotlib.pyplot as plt



def plot_graph(table, x, y, title, xlabel, ylabel, color, kind, legend=True, stacked=False):
    """
    Plots a graph based on the provided table data.

    Parameters:
    - table: DataFrame containing the data to plot.
    - title: Title of the graph.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - color: Color of the plot line.
    - kind: Type of plot (e.g., 'line', 'bar').

    Returns:
    - plotted graph.
    """
    plt.figure(figsize=(10, 6))
    table.plot(x=x, y=y, kind=kind, color=color, legend=legend, stacked=stacked)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    
    fig = plt.gcf()
    return fig
    
def save_graph(graph, filename):
    """
    Saves the plotted graph to a file.

    Parameters:
    - graph: The graph object to save.
    - filename: Name of the file to save the graph as.
    """
    graph.savefig(filename)