import pandas as pd
import numpy as np

def preprocess(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") 
                         for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    def sex_item(x):
        if (x=="male"):
            return 0
        return 1
    
    def embark_item(x):
        if (x=='S'):
            return 0
        return 1

    def nan_item(x):
        if (pd.isna(x)):
            return 0
        return x

    def num_str(x):
        return int(x)

    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)
    df["Sex"] = df["Sex"].apply(sex_item)
    df["Embarked"] = df["Embarked"].apply(embark_item)
    df['Age'] = df['Age'].apply(nan_item)
    df.drop(['Name','Cabin','Ticket_item','Ticket','Ticket_number','PassengerId'],axis=1,inplace=True)
    return df


def read_from_csv():


    train_df = pd.read_csv("../data/titanic/train.csv")
    print(train_df.head(10))

    pre_df = preprocess(train_df)
    print(pre_df.head(10))

    tag = pre_df[['Survived']]
    pre_df.drop('Survived',axis=1,inplace=True)

    return pre_df.to_numpy(),tag.to_numpy()
