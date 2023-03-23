#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:05:01 2022

@author: songsongchangjiang
"""

''' MISM6212 Project data profiling'''
#Changjiang Song

##########################################################################
# Package Preparation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

##########################################################################
# Raw Data Import
df = pd.read_csv("Tourism Survey_March 21, 2022_11.01.csv")
df.shape
df.info()
a = df.loc[[0]]
NA = df.isnull().sum()


'''https://www.statology.org/pandas-rename-columns/'''

##########################################################################
# Rename Columns name

df.rename(columns = {'Q1':'SEX', 'Q2':'AGE', 'Q3':'PROFESSION', 'Q4':'HHINCOME', 'Q5':'VACCINE', 'Q6':'TRANSPORTATION'
                     ,'Q7':'EDUCATION','Q8':'DISTANCE','Q9':'ANXIETY','Q10':'ACCOMMODATION','Q11':'TRAVEL_DURATION',
                     'Q12':'BUDGET','Q13':'COMPANION','Q14':'NECESSITY','Q15':'Questionnaire_satisfaction'}, inplace = True)



df3 = pd.DataFrame(df.apply(lambda x: sum(x.isnull().values), axis = 1)>10,columns = ['TF'])
df3 = df3.loc[df3["TF"] == True]
df3.reset_index(inplace = True)
df3.rename(columns = {'index':'Delete_list'},inplace=True)
Delete_list = df3['Delete_list']
df.drop(Delete_list,inplace = True)

sum(df.apply(lambda x: sum(x.isnull().values),axis = 1)>0)
        
num = list(range(17,79))   
df.iloc[:,num]
df5 = df.iloc[:,num]
df5 = df5.drop(labels=0)

##########################################################################
# Data Formated
'''sex'''
df5['SEX']=df5['SEX'].replace(['男','女'],['Male','Famale'])

'''AGE'''
df5['AGE'] = df5['AGE'].replace(['秘密','50岁','张磊','1953年'],[35,35,35,35])
df5['AGE'] = df5['AGE'].astype('float')
df5['AGE'] = df5['AGE'].astype("Int64")
df5['AGE'] = df5['AGE'].fillna(35)

'''PROFESSION'''
# Checking unique values and count in each column
df5['PROFESSION'].unique()


'''商业：销售，医药代表，客户经理，酒店前台，金融，编辑，职业经理人，打工仔，product manager，普通职员，职员，企业员工，公司职工，文员，公司职员，职工，电子商务，会计，摄影师，管理人员，导购，技术人员

教育业：老师，机构教师，教师，教育工作者

医疗业 ：大健康行业，医生，

工业：工程师，设计师，研发工程师，工人，物流，'建筑类高级工程师，技术员，装修佬，技术人员

农业：农技人员，农民，

科技行业：游戏策划，数字设计工程师，数据分析师，计算机相关

无业：无职业，退休，cl, 主妇，无，无业游民，无业，退休人员

自营业：老板，自由职业，低调土豪，职工及民营股东，法律服务，自由，民营，个体，其它，个体经营，私营它业主，文艺工作者

政府部门：公务员，粮库，公职人员，事业单位人员，干部，政府科员，机关干部，社区工作者，政府工作人员，国企职工，

学生：学生，student，医学生，博士研究生，研究生在读，training，大学学生，'''

'''
business_sector = ['business']*20
education_sector = ['education']*4
engineering_sector = ['engineering']*9
unemployed'['unemployed']*9
print(['Self-operated business']*12)
print(['Government']*10)
print(['Students']*7)
print(engineering_sector)
print(business_sector)
print(education_sector)
medical_sector = ['medical']*2'''

df5['PROFESSION'] = df5['PROFESSION'].replace(['销售','医药代表','客户经理','酒店前台','金融','编辑','职业经理人','打工仔','product manager','普通职员',
                 '职员','企业员工','公司职工','文员','公司职员','职工','电子商务','会计','摄影师','管理人员','导购','技术人员'],
                                              ['business', 'business', 'business', 'business', 'business',
                                               'business', 'business', 'business', 'business', 'business', 
                                               'business', 'business', 'business', 'business', 'business', 
                                               'business', 'business', 'business', 'business', 'business',
                                               'business','business'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['老师','机构教师','教师','教育工作者'],['education', 'education', 'education', 'education'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['大健康行业','医生'],['medical','medical'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['工程师','设计师','研发工程师','工人','物流','建筑类高级工程师','技术员','装修佬','技术人员'],
                                              ['engineering', 'engineering', 'engineering', 'engineering', 'engineering',
                                               'engineering', 'engineering', 'engineering', 'engineering'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['农技人员','农民'],['agriculture','agriculture'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['游戏策划','数字设计工程师','数据分析师','计算机相关'],
                                              ['technology','technology','technology','technology'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['无职业','退休','cl','主妇','无','无业游民','无业','退休人员'],['unemployed','unemployed',
 'unemployed',
 'unemployed',
 'unemployed',
 'unemployed',
 'unemployed',
 'unemployed'])

print(['self-operated business']*13)

df5['PROFESSION'] = df5['PROFESSION'].replace(['老板','自由职业','低调土豪','职工及民营股东','法律服务',
                                              '自由','民营','个体','其它','个体经营','私营它业主','文艺工作者','其他'],
                                              ['self-operated business', 'self-operated business', 'self-operated business', 'self-operated business', 
                                               'self-operated business', 'self-operated business', 'self-operated business', 'self-operated business', 
                                               'self-operated business', 'self-operated business', 'self-operated business', 'self-operated business',
                                               'self-operated business'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['公务员','粮库','公职人员','事业单位人员','干部','政府科员','机关干部','社区工作者','政府工作人员','国企职工'],
                                              ['government',
                                               'government',
                                               'government',
                                               'government',
                                               'government',
                                               'government',
                                               'government',
                                               'government',
                                               'government',
                                               'government'])

df5['PROFESSION'] = df5['PROFESSION'].replace(['学生','student','医学生','博士研究生','研究生在读','training','大学学生','Student'],
                                              ['students', 'students', 'students', 'students', 
                                               'students', 'students', 'students','students'])

    
'''HHINCOME'''
df5.convert_dtypes()

df5['HHINCOME']=df5['HHINCOME'].astype('str')
#['1万-10万','10万-20万','20万-50万','50万-100万','大于100万'] = ['$1500-$15000','$15000-$30000','$30000-$80000','$80000-$150000','Above$150000']

df5['HHINCOME'] =  df5['HHINCOME'].replace(['nan'],['$1500-$15000'])
df5['HHINCOME'] =  df5['HHINCOME'].replace(['$1500-$15000'],['1'])

df5['HHINCOME'] = df5['HHINCOME'].replace(['1万-10万','10万-20万','20万-50万','50万-100万','大于100万'],
                                                                  ['1','2','3','4','5'])
'''vaccine'''
#需要把vaccine里面所有的东西转换为英文的yes or no
df5['VACCINE'] = df5['VACCINE'].replace(['是','否'],['Yes','No'])

'''EDUCATION'''
# '本科' = BACHELOR, '高中及以下' = HighSchool, 研究生及以上 = 'Master', '大专' = 3yearsCollege-3YCollege, '高职高专' = Vocational and Specialized Colleges- VSC
df5['EDUCATION'] = df5['EDUCATION'].replace(['本科','高中及以下','研究生及以上','大专','高职高专'],
                                                                    ['BACHELOR','HighSchool','Master','3YCollege','VSC'])


'''DISTANCE'''
#['最好还是不要接触了' '2米' '无需注意' '1米' '5米' nan] = ['No_Contact','2_meters','insensible','1_meters','5_meters'])
df5['DISTANCE'] = df5['DISTANCE'].astype('str')
df5['DISTANCE'] = df5['DISTANCE'].replace(['最好还是不要接触了','2米','无需注意','1米','5米'],
   ['5','3','1','2','4'])

'''ANXIETY'''
#['新冠就是感冒而已' = insensible '并没有特别焦虑' = somewhat_insensible  '有一定程度的焦虑'=median   'somewhat_anxious '极度焦虑'= very_anxious] 
#df_discriminant['ANXIETY'] = df_discriminant['ANXIETY'].astype('Int64')
df5['ANXIETY'] = df5['ANXIETY'].replace(['新冠就是感冒而已','并没有特别焦虑','有一定程度的焦虑','比较焦虑','极度焦虑'],
    ['insensible','somewhat_insensible','median','somewhat_anxious','very_anxious'])
df5['ANXIETY'] = df5['ANXIETY'].replace(['insensible','somewhat_insensible','median','somewhat_anxious','very_anxious'],
    [1,2,3,4,5])

'''ACCOMMODATION'''
#df5['ACCOMMODATION'].unique()
#'倾向于正规的酒店', '没所谓住宿的形式', '必须要住酒店', '能住酒店就不会选择其他选项', '不愿意住'
#'Tends to Regular Hotel', 'Insensitive', 'Has to be Hotel', ''
df5['ACCOMMODATION'] = df5['ACCOMMODATION'].replace(['不愿意住','倾向于正规的酒店','没所谓住宿的形式','能住酒店就不会选择其他选项','必须要住酒店'],
                                                    ['1','2','3','4','5'])

'''TRAVEL_DURATION'''
#df5['TRAVEL_DURATION'].unique()
#['5-7日', '3-5日', '7-14日', '小于3日', '14日以上']
df5['TRAVEL_DURATION'] = df5['TRAVEL_DURATION'].replace(['5-7日', '3-5日', '7-14日', '小于3日', '14日以上'],
                                                    ['3','2','4','1','5'])
'''BUDGET'''
#df5['BUDGET'].unique()
#'5000元-10000元', '2000元-5000元', '10000元-20000元', '大于20000元','小于2000元']
df5['BUDGET'] = df5['BUDGET'].replace(['5000元-10000元','2000元-5000元','10000元-20000元','大于20000元','小于2000元'],
                                                    ['3','2','4','5','1'])

'''COMPANION'''
#df5['COMPANION'].unique()
#  array(['2人', '3-5人', '5-10人', '单独出行'], dtype=object)
df5['COMPANION'] = df5['COMPANION'].replace(['2人', '3-5人', '5-10人', '单独出行'],
                                                    ['2','3-5','5-10','1'])

'''NECESSITY'''
#df5['NECESSITY'].unique()
#['算是需要', '不需要', '可以但没必要', '一定要有', '挺需要的', nan]
df5['NECESSITY'] = df5['NECESSITY'].replace(['算是需要', '不需要', '可以但没必要', '一定要有', '挺需要的'],
                                                    ['3','1','2','5','4'])

'''Questionnaire_satisfaction'''
df5['Questionnaire_satisfaction'].unique()
#['非常好', '还可以', '不太行', nan]
df5['Questionnaire_satisfaction'] = df5['Questionnaire_satisfaction'].replace(['非常好', '还可以', '不太行'],
                                                    ['Very Good','Good','bad'])
df5['Q25'] = df5['Q25'].replace(['有过','没有'],['Yes','No'])
df5['Q26'] = df5['Q26'].replace(['没有','有'],['No','Yes'])

##########################################################################
# Dealing with Missing Values
col_new = df5.columns.values
col_new_1 = col_new[0:15]

for col in col_new_1:
    df5 = df5.dropna(subset = [col])
    df5 = df5.dropna(subset = [col])

col_Q_1 = df5.columns.values[[15,20,25,30,35,40,45,50,55]]
col_Q_2 = df5.columns.values[[16,21,26,31,36,41,46,51,56]]  
col_Q_3 = df5.columns.values[[17,22,27,32,37,42,47,52,57]]
col_Q_4 = df5.columns.values[[18,23,28,33,38,43,48,53,58]]
col_Q_5 = df5.columns.values[[19,24,29,34,39,44,49,54,59]]

df5 = df5.drop(df5[df5['Questionnaire_satisfaction'] == 'bad'].index)
df5 = df5.drop(df5[df5['HHINCOME'] == 'nan'].index)

for col in col_Q_1:
    df5[col] = df5[col].fillna(1)
for col in col_Q_2:
    df5[col] = df5[col].fillna(2)
for col in col_Q_3:
    df5[col] = df5[col].fillna(3)
for col in col_Q_4:
    df5[col] = df5[col].fillna(4)
for col in col_Q_5:
    df5[col] = df5[col].fillna(5)

df5 = df5.dropna(subset = ['Q26'])
df5.reset_index(drop = True, inplace = True)


cols = df5.columns
for col in cols:
    null_col = df5[col].isnull().sum()
    print(col,'----',null_col)

cols = df5.columns
cols
for col in cols:
    print(col)

    # get a list of unique values
    unique = df5[col].unique()
    print(unique, '\n*************************************\n\n')
    
cols = ['SEX','PROFESSION','VACCINE','TRANSPORTATION','EDUCATION','COMPANION','Questionnaire_satisfaction','Q25','Q26']

##########################################################################
# AsType
for col in cols:
    df5[col] = df5[col].astype('category')

cols = ['HHINCOME', 'DISTANCE', 'ANXIETY', 'ACCOMMODATION', 'TRAVEL_DURATION', 'BUDGET', 'NECESSITY']

for col in cols:
    df5[col] = df5[col].astype('float')
    df5[col] = df5[col].astype('Int64')

cols = ['Q16_1', 'Q16_2', 'Q16_3', 'Q16_4', 'Q16_5', 'Q17_1', 'Q17_2', 'Q17_3',
'Q17_4', 'Q17_5', 'Q18_1', 'Q18_2', 'Q18_3', 'Q18_4', 'Q18_5', 'Q19_1',
'Q19_2', 'Q19_3', 'Q19_4', 'Q19_5', 'Q20_1', 'Q20_2', 'Q20_3', 'Q20_4',
'Q20_5', 'Q21_1', 'Q21_2', 'Q21_3', 'Q21_4', 'Q21_5', 'Q22_1', 'Q22_2',
'Q22_3', 'Q22_4', 'Q22_5', 'Q23_1', 'Q23_2', 'Q23_3', 'Q23_4', 'Q23_5',
'Q24_1', 'Q24_2', 'Q24_3', 'Q24_4', 'Q24_5']

for col in cols:
    df5[col] = df5[col].astype('float')
    df5[col] = df5[col].astype('Int64')

df5.info()
##########################################################################
#16-24所有的问题 1，5兑换，2，4兑换


col_new_2 = col_new[15:60]
for col in col_new_2:
    df5[col] = df5[col].replace([1,2,4,5],[5,4,2,1])
    

##########################################################################
df_discriminant = df5[['SEX', 'AGE', 'PROFESSION', 'HHINCOME', 'VACCINE', 'TRANSPORTATION',
       'EDUCATION', 'DISTANCE', 'ANXIETY', 'ACCOMMODATION', 'TRAVEL_DURATION',
       'BUDGET', 'COMPANION', 'NECESSITY','Q25', 'Q26','Questionnaire_satisfaction']]
df_discriminant = df_discriminant[(df_discriminant['Questionnaire_satisfaction'] =='Very Good') ]
df_PCA_col = df5.columns.values[15:60]
df_PCA = df5[df_PCA_col]

##########################################################################
# Data Export

df5.to_csv("Project_Data_N1.csv", index= True)
df_PCA .to_csv("Project_Data_PCA_N1.csv", index= True)
df_discriminant.to_csv("Project_Data_Discriminant_N2.csv", index= True)



###########################################################################
# Transportation Distribution
from collections import Counter
z = np.array(df5['TRANSPORTATION']).tolist()
num = z.count('自驾,飞机,铁路交通')

z = dict(Counter(z))
print({key:value for key, value in z.items() if value >1})

#自驾排名第一


###########################################################################
# Data without any NA

df5 = df5.dropna(axis=0, how='any',  inplace=False)

df_discriminant = df5[['SEX', 'AGE', 'PROFESSION', 'HHINCOME', 'VACCINE', 'TRANSPORTATION',
       'EDUCATION', 'DISTANCE', 'ANXIETY', 'ACCOMMODATION', 'TRAVEL_DURATION',
       'BUDGET', 'COMPANION', 'NECESSITY','Q25', 'Q26']]

df_PCA_col = df5.columns.values[15:60]
df_PCA = df5[df_PCA_col]

df5.to_csv("Project_Data_Nan.csv", index= True)
df_PCA .to_csv("Project_Data_PCA_Nan.csv", index= True)
df_discriminant.to_csv("Project_Data_Discriminant_Nan.csv", index= True)

###########################################################################
# Addtional Information

df5.columns

'''
Discriminant data rename (EN-CN)
Q1-您的性别是- SEX
Q2-您的年龄是- AGE
Q3-您的职业是- PROFESSION
Q4-您的家庭年收入是- HHINCOME
Q5-请问您是否接种过疫苗- VACCINE
Q6-您更倾向于何种交通工具出行- TRANSPORTATION
Q7-您受教育的程度- EDUCATION
Q8-您认为现在外出旅游时应该和他人保持什么样的社交距离- DISTANCE
Q9-您对新冠疫情的焦虑程度- ANXIETY
Q10-您认为现在外出旅游时更适合住在酒店：（标准三星或四星连锁如：维也纳集团旗下酒店）- ACCOMMODATION
Q11-您认为外出旅游的最佳时长为：- TRAVEL_DURATION
Q12-您认为外出旅游最合适的预算是多少：-BUDGET
Q13-您认为几个人旅游同行是最合适的：- COMPANION
Q14-您是否认为外出旅行时生活里不可以缺少的部分:- NECESSITY
Q15-为了确保调查问卷的可靠性，请您此题选择”非常好“：- Questionnaire_satisfaction
'''
'''
#subset discriminant variables
df_discriminant = df[['SEX','AGE','PROFESSION','HHINCOME','VACCINE','TRANSPORTATION','EDUCATION','DISTANCE','ANXIETY','ACCOMMODATION','TRAVEL_DURATION','BUDGET','COMPANION','NECESSITY','Questionnaire_satisfaction']]
df_discriminant = df_discriminant.drop(labels=0)
#df_discriminant.isnull().sum()

# data visualization for discriminant variables
'''SEX'''
df_discriminant['SEX']=df_discriminant['SEX'].replace(['男','女'],['males','famales'])
sns.histplot(df_discriminant["SEX"],bins=20,color="red",label="Total bill",alpha = 0.5)


'''AGE'''
df_discriminant['AGE'] = df_discriminant['AGE'].replace(['秘密','50岁','张磊','1953年'],[35,35,35,35])
df_discriminant['AGE'] = df_discriminant['AGE'].astype('float')
df_discriminant['AGE'] = df_discriminant['AGE'].astype("Int64")

#df_discriminant = df_discriminant.convert_dtypes()
df_discriminant.info()
#df_discriminant['AGE'].dtype
np.mean(df_discriminant['AGE'])
df_discriminant['AGE'].median()
#mean = 38.2655601659751 =38
np.nanmean(df_discriminant['AGE'])
sns.boxplot(df_discriminant['AGE'], showmeans = True) #boxplot after editing
df_discriminant['AGE'] = df_discriminant['AGE'].fillna(35)
sns.boxplot(df_discriminant['AGE'], showmeans = True) #boxplot after editing

sns.histplot(df_discriminant['AGE']) #histplot

df_discriminant['AGE'].describe()

print(df_discriminant['AGE'].mean())
print(df_discriminant['AGE'].median())
print(df_discriminant['AGE'].mode())

df_discriminant['AGE'].mode()
df_discriminant['AGE'].unique()
df['AGE'].isnull().sum()


NA_dis = df_discriminant['AGE'].isnull().sum()
print(NA_dis)

'''PROFESSION'''
# Checking unique values and count in each column
cols = df_discriminant.columns
cols
for col in cols:
    print(col)

    # get a list of unique values
    unique = df_discriminant[col].unique()
    print(unique, '\n*************************************\n\n')

'''
Student = 学生， 医学生 = 学生， 机构教师 = 教师，退休 = ‘无职业’，数字设计工程师 = 工程师，老板 = 公司CEO
酒店前台 = 公司前台，数据分析师 = 分析师，上市公司ceo，cl？，低调土豪 = 自由职业，打工仔 = 公司职员，职业经理人 = 公司CEO
职工及民营股东 = 自由职业， 博士研究生 = 博士生在读，法律服务 = 法律，自由 = 自由职业，民营 = 自由职业，职员 = 公司职员，
企业员工 = 普通职员，粮库 = 普通职员？，公司职工 = 普通职员，公职人员 = 公务员，事业单位人员 = 公务员，个体 = 自由职业，
建筑类高级工程师 = 工程师， 干部 = 公司职员， training = 实习生，政府科员 = 公务员，计算机相关 = 公司职员，农机人员 = ？
无业 = 无职业，职工 = 公司职工， 个体经营 = 自由职业，装修佬 = 自由职业，摄影师 = 自由职业，管理人员 =公司CEO
私营它业主 = 个体经营， 文艺工作者 = 个体经营，教育工作者 = 教师， 导购 = 销售， 退休人员 = 无职业，技术人员 = ？
'''

'''HHINCOME'''
df_discriminant.convert_dtypes()

df_discriminant['HHINCOME']=df_discriminant['HHINCOME'].astype('str')

df_discriminant['HHINCOME'] = df_discriminant['HHINCOME'].replace(['1万-10万','10万-20万','20万-50万','50万-100万','大于100万'],
                                                                  ['$1500-$15000','$15000-$30000','$30000-$80000','$80000-$150000','Above$150000'])
sns.histplot(df_discriminant['HHINCOME'])
#plt 
plt.hist(df_discriminant['HHINCOME'])
plt.xticks(fontsize=7)

#['20万-50万' '10万-20万' '大于100万' '1万-10万' '50万-100万' nan] 
#$30000-$80000, $15000-$30000, Above$150000, $1500-15000, $80000-$150000


#需要给中文变成英文，人民币转换美元
'''vaccine'''
#需要把vaccine里面所有的东西转换为英文的yes or no
df_discriminant['VACCINE'] = df_discriminant['VACCINE'].replace(['是','否'],['Yes','No'])
sns.histplot(df_discriminant['VACCINE'])

'''Transportation'''
df_discriminant['TRANSPORTATION']=df_discriminant['TRANSPORTATION'].astype('str')
unique = df_discriminant['TRANSPORTATION'].unique()
print(unique, '\n*************************************\n\n')
df_discriminant['TRANSPORTATION'].unique().sum()
sns.histplot( df_discriminant['TRANSPORTATION'])
#这里也遇到问题了，有24个不同组合的交通工具，应该重新进行分类

'''EDUCATION'''
# '本科' = BACHELOR, '高中及以下' = HighSchool, 研究生及以上 = 'Master', '大专' = 3yearsCollege-3YCollege, '高职高专' = Vocational and Specialized Colleges- VSC
df_discriminant['EDUCATION'] = df_discriminant['EDUCATION'].replace(['本科','高中及以下','研究生及以上','大专','高职高专'],
                                                                    ['BACHELOR','HighSchool','Master','3YCollege','VSC'])
sns.histplot( df_discriminant['EDUCATION'])

'''DISTANCE'''
#['最好还是不要接触了' '2米' '无需注意' '1米' '5米' nan] 
df_discriminant['DISTANCE'] = df_discriminant['DISTANCE'].astype('str')
df_discriminant['DISTANCE'] = df_discriminant['DISTANCE'].replace(['最好还是不要接触了','2米','无需注意','1米','5米'],
   ['No_Contact','2_meters','insensible','1_meters','5_meters'])
sns.histplot( df_discriminant['DISTANCE'])

'''ANXIETY'''
#['新冠就是感冒而已' = insensible '并没有特别焦虑' = somewhat_insensible  '有一定程度的焦虑'=median   'somewhat_anxious '极度焦虑'= very_anxious] 
#df_discriminant['ANXIETY'] = df_discriminant['ANXIETY'].astype('Int64')
df_discriminant['ANXIETY'] = df_discriminant['ANXIETY'].replace(['新冠就是感冒而已','并没有特别焦虑','有一定程度的焦虑','比较焦虑','极度焦虑'],
    ['insensible','somewhat_insensible','median','somewhat_anxious','very_anxious'])
df_discriminant['ANXIETY'] = df_discriminant['ANXIETY'].replace(['insensible','somewhat_insensible','median','somewhat_anxious','very_anxious'],
    [1,2,3,4,5])
sns.histplot( df_discriminant['ANXIETY'])
'''


