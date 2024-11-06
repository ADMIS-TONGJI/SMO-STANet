import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def draw_figd(data, name, title):#difference

    fig = sns.heatmap(data, cmap="YlGnBu", xticklabels =False, yticklabels=False)
    #plt.title(title) 
    #fig.set_ylim(0,0.07)
    plt.savefig(name)
    print(name)
    plt.clf()

def draw_figp(data, name, title):#percentage

    fig = sns.heatmap(data, cmap="YlGnBu", xticklabels =False, yticklabels=False)
    #plt.title(title) 
    #fig.set_ylim(0,1.2)
    plt.savefig(name)
    print(name)
    plt.clf()

def draw_fig1(data, name, title):

    fig = sns.heatmap(data, 
                      cmap="YlGnBu", 
                      xticklabels =False, 
                      yticklabels=False,
                      
                      cbar=True,
                      cbar_kws={'label':'Chl-a(mg/m³)',
                                "format":"%.3f"
                                },
                      )
    #plt.title(title) 
    plt.savefig(name)
    print(name)
    plt.clf()


def int_to_date(day, pred_day):
    month = [31,28,31,30,31,30,31,31,30,31,30,31]
    month_run = [31,29,31,30,31,30,31,31,30,31,30,31]
    year = [178,365,366,365,365,
            365,366,365,365,365,
            366,365,365,365,366,
            365,365,365,366,365,
            184]
    pred_day = 7302 - pred_day + day
    
    days = 0
    loc_year = loc_day = loc_month = 0
    for i in range(21):
        days += year[i]
        if days >= pred_day :
            loc_year = i + 2002
            pred_day -= days - year[i]
            break
    
    days = 0
    if loc_year % 4 == 0:#run
        for i in range(12):
            days += month_run[i] 
            if days >= pred_day:
                loc_month = i + 1
                pred_day -= days - month_run[i]
                break
    else:
        for i in range(12):
            days += month[i] 
            if days >= pred_day:
                loc_month = i + 1
                pred_day -= days - month_run[i]
                break
    
    loc_day = pred_day
    print(str(loc_year) + '-' + str(loc_month) + '-' + str(loc_day))
    return str(loc_year) + '-' + str(loc_month) + '-' + str(loc_day)


def visual(name, pred_len, pred_day, data_path):
    
    date = int_to_date(0, pred_day)
    c = np.load("./results/" + name + "/real_prediction.npy")
    pred = c[0]
    folder_path = './visual/' + name +'/'
    file = pd.read_csv(data_path)
    cols_data = file.columns[2:2306]
    df_data = file[cols_data]
    fill = np.array(df_data)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    loss = 0
    for day in range(pred_len):
        tmp1 = pred[day].reshape(48,48)
        '''for i in range(48):
            for j in range(48):
                tmp[i][j] = 0 if tmp[i][j] < 0 else tmp[i][j]'''
        
        tmp2 = fill[7303 - int(pred_day) - int(pred_len) + day].reshape(48,48)

        date = int_to_date(day, pred_day+pred_len-1)

        draw_fig1(tmp1,folder_path + str(day) + ".png", date + "_Prediction")
        draw_figd(np.abs(tmp1 - tmp2),folder_path + "residue" + str(day) + ".png", date + "_Difference")
        
        draw_figp(np.abs(tmp1 - tmp2) / tmp2,folder_path + "percent" + str(day) + ".png", date + "_Difference_Percentage")
        draw_fig1(tmp2,folder_path + "real" + str(day) + ".png", date + "_Real")
        print(folder_path)
        loss += np.sum(np.abs(tmp1 -tmp2))/2304 

    result = open(folder_path + "/visual_metric" ,"w")
    result.write('average loss:' + str(loss/pred_len))

    
