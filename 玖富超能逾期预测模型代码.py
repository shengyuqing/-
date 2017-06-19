# -*- coding:UTF-8 BOM -*-  
"""
Created on Tue Nov 15 09:55:23 2016

@author: Administrator
"""

import graphlab as gl
import pandas as pd
#import graphlab.aggregate as agg
'''
import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library
#%matplotlib inline 
# A magic command that tells matplotlib to render figures as static images in the Notebook.
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
#sns.axes_style()，可以看到是否成功设定字体为微软雅黑。
import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).
#sns.set_style('whitegrid') # One of the five seaborn themes
import warnings
warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg
from scipy import stats, linalg
'''
'''
import sys
# print sys.getdefaultencoding()
# ipython notebook中默认是ascii编码 
reload(sys)
sys.setdefaultencoding('utf8') 
'''
#import UTF8_GBK
#UTF8_GBK.GBK_2_UTF8('data.csv','utf8.csv')



data_zhapian = gl.SFrame.read_csv('zhapian.csv',
                                  column_type_hints={
                                  'id':str,
                                  'first_name':str,
                                  'shop_name':str,
                                  'product_series_name':str,                                         
                                  'loan_money':float,
                                  'user_evaluate':str,
                                  'periods':int,
                                  'interest_rate':float,
                                  'amount':float,
                                  'down_payment_amount':float,
                                  '年龄':int,   
                                  'repayment_bank':str,
                                  'repayment_city':str,
                                  'sa_remark':str,
                                  'self_repay_bank':str,
                                  'monthly_premium_rate':float,
                                  'merchant_type_code':str,
                                  'major_contact_position':str,                                  
                                  'CONTR_ANNUAL_INTEREST_RATE':float,
                                  'reserve_rate':float,
                                  'buy_insurance':str,
                                  'check_return':str,
                                  'payment_bank':str,
                                  'payment_city':str,                                 
                                  'parent_id':str,
                                  'provincename':str,
                                  'store_name':str,
                                  'merchant_type_code':str,
                                  'major_contact_position':str,
                                  'product_series_name1':str,
                                  'product_category':str,
                                  'customer_sex':str,
                                  'children_number':int,                                  
                                  'first_grade_level':str,
                                  'sec_grade_level':str,                                  
                                  'degree':str,
                                  'marry':str,
                                  'addre_regist':str,
                                  'live_condition':str,
                                  'household_prov':str,
                                  'household_city':str,
                                  'household_town':str,
                                  'household_postcode':str,
                                  'live_prov':str,
                                  'live_city':str,
                                  'live_town':str,
                                  'live_postcode':str,
                                  'monthly_income':int,
                                  'other_income':int,
                                  'family_income':int,
                                  'monthly_spending':int,
                                  'industry':str,
                                  'unittype':str,
                                  'department':str,
                                  'duty':str,
                                  'is_social_sec':str,
                                  'unit_addre':str,
                                  'unit_city':str,
                                  'unit_town':str,
                                  'career':str,
                                  'unit_worktime':int,
                                  'unit_postcode':str,
                                  'post_state':str,
                                  'first_worktime':int,
                                  'contract_type':str,
                                  'contract_address':str,
                                  'contract_property':str,
                                  'contract_unit':str,
                                  'deadline_status':str,
                                  'result':str
                                  })

data_zhengmian = gl.SFrame.read_csv('zhengmian.csv',
                           column_type_hints={
                                  'id':str,
                                  'first_name':str,
                                  'shop_name':str,
                                  'product_series_name':str,                                         
                                  'loan_money':float,
                                  'user_evaluate':str,
                                  'periods':int,
                                  'interest_rate':float,
                                  'amount':float,
                                  'down_payment_amount':float,
                                  '年龄':int,   
                                  'repayment_bank':str,
                                  'repayment_city':str,
                                  'sa_remark':str,
                                  'self_repay_bank':str,
                                  'monthly_premium_rate':float,
                                  'merchant_type_code':str,
                                  'major_contact_position':str,                                  
                                  'CONTR_ANNUAL_INTEREST_RATE':float,
                                  'reserve_rate':float,
                                  'buy_insurance':str,
                                  'check_return':str,

                                  'payment_bank':str,
                                  'payment_city':str,
                                  
                                  'parent_id':str,
                                  'provincename':str,
                                  'store_name':str,
                                  'merchant_type_code':str,
                                  'major_contact_position':str,
                                  'product_series_name1':str,
                                  'product_category':str,
                                  'customer_sex':str,
                                  'children_number':int,
                                  
                                  'first_grade_level':str,
                                  'sec_grade_level':str,
                                  
                                  'degree':str,
                                  'marry':str,
                                  'addre_regist':str,
                                  'live_condition':str,
                                  'household_prov':str,
                                  'household_city':str,
                                  'household_town':str,
                                  'household_postcode':str,
                                  'live_prov':str,
                                  'live_city':str,
                                  'live_town':str,
                                  'live_postcode':str,
                                  'monthly_income':int,
                                  'other_income':int,
                                  'family_income':int,
                                  'monthly_spending':int,
                                  'industry':str,
                                  'unittype':str,
                                  'department':str,
                                  'duty':str,
                                  'is_social_sec':str,
                                  'unit_addre':str,
                                  'unit_city':str,
                                  'unit_town':str,
                                  'career':str,
                                  'unit_worktime':int,
                                  'unit_postcode':str,
                                  'post_state':str,
                                  'first_worktime':int,
                                  'contract_type':str,
                                  'contract_address':str,
                                  'contract_property':str,
                                  'contract_unit':str,
                                  'deadline_status':str,
                                  'result':str
                                  })

data_zhapian_t = data_zhapian.column_types()
data_zhengmian_t = data_zhengmian.column_types()
colum_names = data_zhapian.column_names()

#for i in range(len(data_zhapian_t)):
#    print (colum_names[i], data_zhapian_t[i], data_zhengmian_t[i])  


data = data_zhapian.append(data_zhengmian)

#舍弃的特征
drop_columns = ['id',
                'app_state',
                'user_evaluate',
                'code',
                'name',
                'GroupID',
                'firm_name',
                'shop_name',
                'wo_build_time',
                #'sa_remark',
                'is_upload_attach',
                'app_status_name',  
                'name.1',
                'reserve_rate' ,
                #'interest_rate',
                #'product_name',
                #'product_series_name',
                #'product_series_name.1'
]
#转换有自然语言的特征
data['sa_remark'] = data['sa_remark'].apply(lambda x : +1 if x != '' else 0)

data.remove_columns(drop_columns)

#重新命名特征
data.rename({'年龄':'age',
             })
colum_names = data.column_names()

#随机排列数据集
data = gl.toolkits.cross_validation.shuffle(data, random_seed=1)
#转换输出特征
data['result'] = data['result'].apply(lambda x : 'bad' if x == '1' else 'good')
#确定特征列
features = data.column_names()
features_type = data.column_types()
features.remove('result')

#确定目标列
target = 'result'

#让通过和拒绝的数据集接近
#拒绝的标1，通过的标0
rej_data_raw = data[data[target] == 'bad']
pass_data_raw = data[data[target] == 'good']
print "通过总量  : %s" % len(pass_data_raw)
print "拒绝总量 : %s" % len(rej_data_raw)

percentage = float(len(rej_data_raw))/len(pass_data_raw)
print percentage
rej_data = rej_data_raw
pass_data = pass_data_raw.sample(percentage, seed=1)

print "通过总量  : %s" % len(pass_data)
print "拒绝总量 : %s" % len(rej_data)

input_data = pass_data.append(rej_data)
input_data = gl.toolkits.cross_validation.shuffle(input_data, random_seed=1)


#划分数据集，训练集、验证集和测试集
train_valid_data, test_data = input_data.random_split(0.8, seed=1)
train_data, valid_data = train_valid_data.random_split(0.8, seed=1)



#1、选用决策树
model1 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 1)
                               
model2 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 2) 
model3 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 3) 
model6 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 6)                                 
'''                                
model1 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 1)
model2 = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 2) 

                                
model150= gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 1000)
'''
#print model150.evaluate(valid_data)['accuracy']

#图像化展示决策树模型
model3.show(view="Tree")
'''
print model92.evaluate(test_data, metric='precision')
print model92.evaluate(test_data, metric='recall')
print  model100.evaluate(valid_data)['accuracy']
'''
'''
'''
#classes = model.classify(valid_data)     
#print  classes  
'''        
acc1 = model1.evaluate(valid_data)['accuracy']
acc2 = model2.evaluate(valid_data)['accuracy']
acc6 = model6.evaluate(valid_data)['accuracy']
acc92 = model92.evaluate(valid_data)['accuracy']
print acc1
print acc2
print acc6
print acc92
model1.classes(valid_data)
model6.show(view="Tree", vlabel_hover=True)  
'''

#打印出1层到100层决策树分别在验证集上的准确率，来选取层数
'''
best_layer = 0
best_acc = 0
num_layer = 0
acc_list =[]
for i in range(100):
    num_layer  = i+1
    model = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = i+1)
    acc1 = model.evaluate(train_data)['accuracy']                             
    acc2 = model.evaluate(valid_data)['accuracy'] 
    acc_list.append((num_layer,acc1,acc2))
print acc_list
'''

#打印出不同最小损失对应的决策树在验证集上的准确率，来选取最小损失
'''
#num_layer = 0
loss_list1 = []
num_list = [0, 0.1,0.2, 0.5 , 1, 1.5, 1.8 , 100 ,200]
for num_loss in num_list:
    
    model = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 11,min_loss_reduction=i)
    acc1 = model.evaluate(train_data)['accuracy']                             
    acc2 = model.evaluate(valid_data)['accuracy'] 
    loss_list1.append((num_loss,acc1,acc2))
print loss_list1

print (best_layer, best_acc)
model = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 3)
                                
                                
model.show(view="Tree")    
print  model.evaluate(test_data)['accuracy']


 
model_1 = gl.logistic_classifier.create(train_data,target='result',features=features,l2_penalty=10000)
result = model.evaluate(valid_data)['accuracy']
print result 
'''


#2、选用adaboost模型
'''
model_1 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 1,max_depth = 70)
model_5 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 5,max_depth = 70)        
model_10 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 10,max_depth = 70)  
model_15 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 15,max_depth = 70)
model_19 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 19,max_depth = 70) 
model_20 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 20,max_depth = 70) 
model_30 = gl.boosted_trees_classifier.create(train_data, validation_set=None, 
        target = 'result', features = features, max_iterations = 30,max_depth = 70) 
              
print model_1.evaluate(valid_data,metric='accuracy')
print model_5.evaluate(valid_data,metric='accuracy')
print model_10.evaluate(valid_data,metric='accuracy')
print model_15.evaluate(valid_data,metric='accuracy')
print model_19.evaluate(valid_data,metric='accuracy')
print model_20.evaluate(valid_data,metric='accuracy')
print model_25.evaluate(valid_data,metric='accuracy')
print model_30.evaluate(valid_data,metric='accuracy')

#选定每棵树的层数，然后打印出1到100棵树的adaboost在验证集上准确率
model = gl.decision_tree_classifier.create(train_data, validation_set=None,
                                target = 'result', features = features,max_depth = 4) 

test_data['predictions'] = model100.predict(test_data,output_type='probability')
test_data['predic_class'] = model100.predict(test_data,output_type='class')
print test_data['result','predictions','predic_class']
a = test_data[test_data['result'] != test_data['predic_class']].num_rows()
print float(a)/11604
model.show(view="Tree") 




print model100.evaluate(data, metric='precision')
print model100.evaluate(data, metric='recall')
#画出ROC曲线判断模型的效果
roc = model_5.evaluate(test_data, metric='roc_curve')
'''

