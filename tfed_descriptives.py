import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

# set plotting style
plt.style.use('ggplot')
    
# helper function to plot histograms of dependent variables
def depvar_hists(data, keys):
    
    # how many histograms to plot
    n_hists = len(keys)
    
    # determine shape of subplot grid
    dim2 = math.ceil(np.sqrt(n_hists))
    dim1 = math.floor(np.sqrt(n_hists)) + int(n_hists > math.floor(np.sqrt(n_hists)) * dim2)
    
    # initialize subplots        
    fig, axs = plt.subplots(dim1, dim2, sharex=True,sharey=True, constrained_layout=True)
    fig.suptitle('Turbine Lifespans', fontsize=22)
    
    # initialize list of max cycle by unit 
    max_cycles = [None] * n_hists
    
    # fill max cycles list
    i = 0
    for table in keys:
        max_cycles[i] = np.array(data[table].groupby('unit').max()['cycle'])
        i += 1
    
    # fill figure
    i = 0
    # if just one histogram needs to be plotted
    if dim1 == dim2 == 1:
        axs.hist(max_cycles, density=True)
        axs.set_title(keys[i], fontsize=16)
        axs.set(xlabel='Cycles', ylabel='Frequency')
    # if only one row of subplots needs to be plotted
    elif dim1 == 1 and dim2 > 1:
        for dii in range(0, dim2):
            if i < len(max_cycles):
                axs[dii].hist(max_cycles[i], density=True)
                axs[dii].set_title(keys[i], fontsize=16)
                if dii == 0:
                    axs[dii].set(xlabel='Cycles', ylabel='Frequency')
                else:
                     axs[dii].set(xlabel='Cycles')
                i += 1
    # if two dimensional grid of subplots is to be plotted
    else:
        for di in range(0, dim1):
            for dii in range(0, dim2):
                if i < len(max_cycles):
                    axs[di, dii].hist(max_cycles[i], density=True, label = keys[i])
                    axs[di, dii].set_title(keys[i], fontsize=16)
                    if di == dim1-1 and dii == 0:
                        axs[di, dii].set(xlabel='Cycles', ylabel='Frequency')
                    elif di != dim1-1 and dii == 0:
                        axs[di, dii].set(ylabel='Frequency')
                    elif di == dim1-1 and dii != 0:
                        axs[di, dii].set(xlabel='Cycles')
                    i += 1
                else:
                    break
    
    # store figure
    #plt.tight_layout()
    plt.savefig('descriptives1_depvar_hists.pdf')
    plt.close()

# helper function to plot sensor means conditional on how many cycles to failure
def sensor_means_up_to_failure(df):
    col_filter = [col for col in df if col.startswith(('unit', 'cycle', 'sensor'))]
    sensors = [col for col in df if col.startswith('sensor')]
    df = df[col_filter]
    df = df.assign(max_cycle=df.groupby('unit')['cycle'].transform('max'))
    df['cycles_to_failure'] = df['max_cycle'] - df['cycle']
    
    # determine start of x-axes of plots
    shortest_y = min(df['max_cycle'])
    
    # initialize plot data
    plot_data = pd.DataFrame({'cycles_to_failure':np.arange(0, max(df['cycle']))})
    
    for i in sensors:
        plot_data[i] = df.groupby('cycles_to_failure')[i].mean()
        
    plot_data = plot_data.loc[plot_data['cycles_to_failure'] <= shortest_y]
    
    fig, axs = plt.subplots(int(len(sensors)/3), 3, figsize=(11.7,8.27), 
                            sharex=True, constrained_layout=True)
    fig.suptitle('Sensor Means up to Failure', fontsize=22)
    for l in [0,1,2]:
        i = 0
        for s in range(l, len(sensors), 3):
            axs[i,l].set_xlim(shortest_y, 0)
            axs[i,l].set_title(sensors[s], fontsize=10)
            if s > len(sensors)-4: # last row gets x-labels
                axs[i,l].set(xlabel='Cycles to Failure')
            axs[i,l].plot(plot_data['cycles_to_failure'], plot_data[sensors[s]])
            i += 1
    
    #plt.tight_layout()
    fig.savefig('descriptives2_sensor_means_up_to_failure.pdf')
    plt.close()

# helper function to plot correlation heat map of independent variables
def indepvar_cmap(df):
    col_filter = [col for col in df.columns if df[col].std() != 0 and \
                  (col.startswith('op') or col.startswith('sensor'))]
    corr_df = df[col_filter]
    print("The following variables have been dropped from correlation heatmap due to zero variance:",\
          "\n", [col for col in df.columns if df[col].std() == 0 and \
                 (col.startswith('op') or col.startswith('sensor'))])
    #sns.set(rc={'figure.figsize':(11.7,8.27)})
    fig, cmap = plt.subplots(constrained_layout=True)
    fig.suptitle('Correlation Heatmap of Features', fontsize=22)
    sns.heatmap(corr_df.corr(), cmap="RdBu_r")
    #cmap_fig = cmap.get_figure()
    #cmap_fig.savefig('descriptives3_indepvar_cmap.pdf')
    fig.savefig('descriptives3_indepvar_cmap.pdf')
    plt.close()