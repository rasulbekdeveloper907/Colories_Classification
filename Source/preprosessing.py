
import pandas as pd

class Cleaner:
    def __init__(self, df):
        self.df = df

    def tozala(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self

    def get_df(self):
        return self.df
    

    
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Encoder:
    def __init__(self, df, target_col=None):
        """
        df : pandas DataFrame
        target_col : target ustun nomi (kodlanmasligi kerak bo‘lgan)
        """
        self.df = df.copy()
        self.target_col = target_col
        self.encoder = LabelEncoder()

    def encodla(self):
        for col in self.df.columns:
            # 🎯 Target ustunni tashlab o‘tish
            if col == self.target_col:
                continue

            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 5:
                    # 🌈 Kam qiymatlar uchun one-hot encoding
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                else:
                    # 🔢 Ko‘p qiymatlar uchun label encoding
                    self.df[col] = self.encoder.fit_transform(self.df[col].astype(str))
        return self

    def get_df(self):
        return self.df

    
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Scaler:
    def __init__(self, df, target_col=None):
        """
        df: DataFrame
        target_col: Target ustun nomi (str), scalingdan chiqarib tashlanadi
        """
        self.df = df.copy()
        self.target_col = target_col
        self.scaler = StandardScaler()

    def scaling_qil(self):
        # Target ustunni ajratamiz (agar ko‘rsatilgan bo‘lsa)
        if self.target_col and self.target_col in self.df.columns:
            target = self.df[self.target_col]
            df_temp = self.df.drop(columns=[self.target_col])
        else:
            target = None
            df_temp = self.df

        # Faqat raqamli ustunlarni tanlaymiz
        numeric_cols = df_temp.select_dtypes(include=['int64', 'float64']).columns

        # Skalalash
        df_temp[numeric_cols] = pd.DataFrame(
            self.scaler.fit_transform(df_temp[numeric_cols]),
            columns=numeric_cols,
            index=df_temp.index
        )

        # Agar target bor bo‘lsa, uni qaytadan qo‘shamiz
        if target is not None:
            df_temp[self.target_col] = target

        # Yangi DataFrame’ni saqlaymiz
        self.df = df_temp
        return self

    def get_df(self):
        return self.df

