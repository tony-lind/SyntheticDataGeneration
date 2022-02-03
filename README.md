The code is the basis for the paper: "Low dimensional synthetic data generation for improving data driven prognostic models" 
In order to re-run the experiment in the paper or just use the some of the synethtic data generation methods the following steps needs to be done:

Install all packages needed for running the code

In all files, adjusting the codes behaviour is done by '#' to hide certain parts of the code.  
Files:\
syntetic_data_from_embedding.py - creates synthetic data and writes the synthetic examples to disc\
line 27-32 define where to write your data, change this according to your preference\
line 34 update path to where you have put the embeddings data, i.e. embeddings.txt\
line 155 de-comment if you want to use KDE or KDE with check\
line 169 de-comment if you want to use KDE with check\
line 181-190 de-comment the method you want to run

evaluation_baseline_kfold.py - runs experiment with the baseline method, i.e, no usage of sythetic data and write result to file named 'result_no_syn'.\
line 193-195 update path to where you have put the embeddings data, i.e. embeddings.txt

evaluation_synthetic_kfold.py - runs experiment with the synthetic data generated and write result to file.\
line 193-195 path were you put the synthetic data and embeddings.txt

cost_size_plot - plot figure 4 in the paper, i.e., cost-size result for all methods\
line 15-21 update path to where you have the results for each method

learning_curves.py - plot figure 5 in the paper, i.e., learning curve for basline methods and SMOTE\
line 26-29 update path to where you have put the embeddings data, i.e. embeddings.txt\
As is the learning curve for basline data will be plotted. To plot synthetic data:\
line 35 comment away\
line 37-39 de-comment\ 
line 47-49 comment away\
line 50-52 de-comment

t_sne_baseline_plotting.py - plot figure 3\
line 54-56 update path to where you have put the embeddings data, i.e. embeddings.txt

t_sne_synthetic_plotting.py - plot figure 6\
line 64-66 update path to where you have put the embeddings data, i.e. embeddings.txt\
line 77-79 update to path where your synthetic data are

Best regards,

Anonymus
