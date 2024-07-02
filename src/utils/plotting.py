import matplotlib.pyplot as plt


def configure_plotting():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({'font.size': 10})  # You can change the number to your desired font size
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.title_fontsize'] = 10

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'times'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # For math support
