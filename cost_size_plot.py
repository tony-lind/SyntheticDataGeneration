"""
Description:
Using result measurements written to file this file creates figures and plots 

Input: textfiles of results
Output: figures and results

@author: tlim3c
"""
import matplotlib.pyplot as plt
import pickle

#Set this to the path you want to write your synthetic data
print("?-read in data")
path = 'E:/SynologyDrive/programmering/eclipse-workspace/TapNet/plot/high/plot_data/'
data1 = '_syn1_'
data2 = '_syn2_'
data3 = '_syn5_'
data4 = '_syn3_'
data5 = '_syn4_'
data6 = '_syn6_'
data_baseline = pickle.load(open(path + 'result_no_syn','rb'))

size_list = [100, 200, 400, 600, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000]

syn1_data, syn2_data, syn3_data, syn4_data, syn5_data, syn6_data = {}, {}, {}, {}, {}, {}
for size in size_list:
    syn1_data[size] = pickle.load(open(path + 'result' + data1 + str(size),'rb'))
    syn2_data[size] = pickle.load(open(path + 'result' + data2 + str(size),'rb'))
    syn3_data[size] = pickle.load(open(path + 'result' + data3 + str(size),'rb'))
    syn4_data[size] = pickle.load(open(path + 'result' + data4 + str(size),'rb'))
    syn5_data[size] = pickle.load(open(path + 'result' + data5 + str(size),'rb'))
    syn6_data[size] = pickle.load(open(path + 'result' + data6 + str(size),'rb'))
print('done reading in data')

print('transform data for plotting')
syn1_y_plot = []
syn2_y_plot = []
syn3_y_plot = []
syn4_y_plot = []
syn5_y_plot = []
syn6_y_plot = []
y_min, y_max = 9999999999, 0
for size in size_list:
    syn1_y_val = syn1_data[size].get('cost_mean')
    syn1_y_plot.append(syn1_y_val) 
    if syn1_y_val > y_max:
        y_max = syn1_y_val
    if syn1_y_val < y_min:
        y_min = syn1_y_val    
    syn2_y_val = syn2_data[size].get('cost_mean')
    syn2_y_plot.append(syn2_y_val) 
    if syn2_y_val > y_max:
        y_max = syn2_y_val
    if syn2_y_val < y_min:
        y_min = syn2_y_val
    syn3_y_val = syn3_data[size].get('cost_mean')
    syn3_y_plot.append(syn3_y_val) 
    if syn3_y_val > y_max:
        y_max = syn3_y_val
    if syn3_y_val < y_min:
        y_min = syn3_y_val
    syn4_y_val = syn4_data[size].get('cost_mean')
    syn4_y_plot.append(syn4_y_val) 
    if syn4_y_val > y_max:
        y_max = syn4_y_val
    if syn4_y_val < y_min:
        y_min = syn4_y_val
    syn5_y_val = syn5_data[size].get('cost_mean')
    syn5_y_plot.append(syn5_y_val) 
    if syn5_y_val > y_max:
        y_max = syn5_y_val
    if syn5_y_val < y_min:
        y_min = syn5_y_val
    syn6_y_val = syn6_data[size].get('cost_mean')
    syn6_y_plot.append(syn6_y_val) 
    if syn6_y_val > y_max:
        y_max = syn6_y_val
    if syn6_y_val < y_min:
        y_min = syn6_y_val    
        
x_plot = size_list
#syn1_p = 
plt.hlines(y= data_baseline.get('cost_mean'), xmin=size_list[0], xmax=size_list[-1], label="no synthetic data")
plt.plot(x_plot, syn1_y_plot, label='uniform_interpolation')
#syn2_p = 
plt.plot(x_plot, syn2_y_plot, label='exponential_interpolation')
#syn3_p = 
plt.plot(x_plot, syn3_y_plot, label='stepwise_uniform_interpolation')
#syn4_p = 
plt.plot(x_plot, syn4_y_plot, label='KDE')
plt.plot(x_plot, syn5_y_plot, label='KDE_w_check')
plt.plot(x_plot, syn6_y_plot, label='SMOTE')

plt.legend() #handles=[syn1_p, syn2_p, syn3_p, syn4_p])
plt.title('cost-synthetic size visualization')
plt.xlabel('Size')
plt.ylabel('Cost')
ymin = y_min
ymax = y_max
#plt.vlines(x= best_threshold, ymin=ymin, ymax=ymax, colors="red", label="best threshold")

#plt.show()
plt.savefig(path + 'cost_size')

print('done plotting')
