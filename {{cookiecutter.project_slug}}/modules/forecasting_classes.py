import gcsfs
import queries
from google.cloud import bigquery
import pandas as pd
import os
import holidays
from google.cloud import storage
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
import string
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import lightgbm as lgb
import pyarrow.parquet as pq
from google.cloud import storage
import string
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
import pickle

# adicionar funcoes para listar os arquivos
class SaveToCloud():
    def __init__(self, project_name, bucket_name):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.fs = gcsfs.GCSFileSystem(project=project_name)
        self.client = bigquery.Client(project=project_name)
        # posso criar mais um self com path

    def save_parquet_to_gcp(self, df, prefixo='', destino=''):
        client_bucket = storage.Client()
        bucket = client_bucket.get_bucket(self.bucket_name)
        parquet_file_path = f'{prefixo}.parquet'
        df.to_parquet(parquet_file_path)
        blob = bucket.blob(f'{destino}/{prefixo}.parquet')
        with open(parquet_file_path, 'rb') as file:
            blob.upload_from_file(file)
        print('Arquivo salvo com sucesso')
        os.remove(parquet_file_path)
        print('Arquivo temporário removido com sucesso')

    def save_csv(self, df, file_name): #acho que tem que editar essa func
        file_path = f"{file_name}.csv"
        with self.fs.open(f"{self.bucket_name}/{file_path}", "wb") as f:
            df.to_csv(f, index=False)
        print(f"CSV file '{file_path}' saved to GCP.")
        
        #bucket_name,
    def save_model_to_gcs(self,model, path, blob_name):
        # Serialize the model
        serialized_model = pickle.dumps(model)

        # Create a GCS client
        #storage_client = storage.Client()
        client_bucket = storage.Client()
        # Get the bucket
        #bucket = storage_client.bucket(bucket_name)
        bucket = client_bucket.get_bucket(self.bucket_name)
        # Create a blob (file) in the specified path within the bucket
        blob = bucket.blob(f"{path}/{blob_name}")

        # Upload the serialized model to the blob
        blob.upload_from_string(serialized_model)    

""
class Dataset:
    def __init__(self,client):
        self.client = client.client
        
    def fetch_data(self, client, loja):
        # Fetch store data using the provided client and store identifier
        query_all_data = queries.query_all_data(loja=loja)
        query_job = client.query(query_all_data)   
        results = query_job.to_dataframe().clean_names()
        return results
# colocar outras métricas aqui de metadados
    def calculate_sales_stats(self, store_data):
        store_ini = store_data['store_ini'].drop_duplicates().tolist()[0]
        max_date = store_data['date_key'].max()
        min_date = store_data['date_key'].min()
        distinct_items = len(store_data['item_desc'].drop_duplicates())
        distinct_groups = len(store_data['group_desc'].drop_duplicates())
        distinct_classes = len(store_data['group_desc'].drop_duplicates())
        distinct_sub_classes = len(store_data['sub_class_desc'].drop_duplicates())
        distinct_brand_type_desc = len(store_data['brand_type_desc'].drop_duplicates())
        total_sales = len(store_data)
        total_sales_qty = store_data['sales_qty_units_sold'].sum()
        promo_flag_percentage = (store_data['promo_flag'] == 1).mean() * 100
        
        return pd.DataFrame({
    'store_ini': store_ini,
    'max_date': max_date,
    'min_date': min_date,
    'distinct_items': distinct_items,
    'distinct_groups': distinct_groups,
    'distinct_classes': distinct_classes,
    'distinct_sub_classes': distinct_sub_classes,
    'distinct_brand_type_desc': distinct_brand_type_desc,
    'total_sales': total_sales,
    'total_sales_qty': total_sales_qty,
    'promo_flag_percentage': promo_flag_percentage
},index=[0])

    def na_counts_df(self,store_data):
    
        # Count NA values for each column
        na_counts = store_data.isna().sum()

        # Calculate proportions
        total_rows = len(store_data)
        na_proportions = na_counts / total_rows

        # Create DataFrame with NA counts and proportions
        na_counts_df = pd.DataFrame({'Column': na_counts.index, 'NA_Count': na_counts.values, 'Proportion': na_proportions.values}).sort_values(by='NA_Count', ascending=False)
    
        return na_counts_df
    
    def add_holidays(self, store_data):
    
        br_holidays = holidays.BR()
        
        store_data['date_key'] = pd.to_datetime(store_data['date_key'], format='%Y-%m-%d')
        store_data['holidays'] = [i in br_holidays for i in store_data['date_key']]
        store_data['holidays'] = store_data['holidays'].astype(int)
        counter_holidays_before = np.zeros(len(store_data))
        counter_holidays_after = np.zeros(len(store_data))
        for i, value in enumerate(store_data['holidays']):
            if value:
                for j in range(8):
                    counter_holidays_before[i-j] = j
                    counter_holidays_after[i+j-1] = j
        store_data['counter_holidays_before'] = counter_holidays_before
        store_data['counter_holidays_after'] = counter_holidays_after    
        
        return store_data
    
    def format_date_data(self,store_data):
        store_data['date_key'] = pd.to_datetime(store_data['date_key'])  
        store_data['day_of_week'] = store_data['date_key'].dt.day_name() # tá redundante
        store_data['week_of_month'] = store_data['date_key'].dt.day // 7 + 1  # semana do mes
        store_data['day_of_month'] = store_data['date_key'].dt.day
        store_data['month_of_year'] = store_data['date_key'].dt.month
        
        store_data['delta_days_promo'] = (store_data['max_end_date'] - store_data['min_start_date']).dt.days
        store_data.drop(columns=['day_of_week_sql']) # tirando coluna redundante
        return store_data
        
    def get_inputations(self,store_data,window=7,min_periods=1,grouping_var='item_desc',inputation_var='sales_avg_unit_price_with_tax',output_column='inputed_price'):
        
        df_inputation = store_data.groupby(grouping_var)[inputation_var].rolling(window=7, min_periods=1).mean().reset_index()
        df_inputation = df_inputation.rename(columns={inputation_var: f"{output_column}{window}"}).drop(columns=[grouping_var])
        df_final = pd.merge(store_data, df_inputation, left_index=True, right_on='level_1', how='left')
        df_final.drop(columns=['level_1'],inplace=True)
        return(df_final)
        
    def process_store(self, client, loja):
        # Fetch store data
        data = self.fetch_data(client, loja)
        data = self.add_holidays(data)
        data = self.format_date_data(data)
        data = self.get_inputations(data,output_column='inputed_price_item')
        data = self.get_inputations(data,window=30,output_column='inputed_price_item')
        data = self.get_inputations(data,grouping_var='class_desc',output_column='inputed_price_class')
        data = self.get_inputations(data,window=30,grouping_var='class_desc',output_column='inputed_price_class')
        data = self.get_inputations(data,grouping_var='group_desc',output_column='inputed_price_group')
        data = self.get_inputations(data,window=30,grouping_var='class_desc',output_column='inputed_price_group')
        data = self.get_inputations(data,grouping_var='item_desc',output_column='inputed_sales',inputation_var='sales_qty_units_sold') # incluindo input de vendas
        data = self.get_inputations(data,window=30,grouping_var='item_desc',output_column='inputed_sales',inputation_var='sales_qty_units_sold') # idem
        
        na_counts_df = self.na_counts_df(data)
        # Calculate date key statistics
        sales_stats = self.calculate_sales_stats(data)
        
        
        return {
            'sales_data': data,
            'sales_stats': sales_stats,
            'na_count': na_counts_df
        }

""
# vale incluir algumas vars aqui pro split
# split = 0.7 e self.split = split, por exemplo
class Pre_processing():
    def __init__(self, data):
        self.data = data
    
    # dá para mudar esse nome
    def process_split_format(self,data,split=.80,target='sales_qty_units_sold'):
        data.sort_values(by=['date_key'], ascending=True, inplace=True)
        data.dropna(subset=['sales_qty_units_sold'],inplace=True)
        train_set, test_set= np.split(data, [int(split *len(data))])
        
        return train_set,test_set
    
    def split_by_date(self, train_set, test_set):
        max_date = train_set['date_key'].max().strftime('%Y-%m-%d')
        train_set = train_set[train_set['date_key'] < max_date]
        test_set = test_set[test_set['date_key'] >= max_date]
        return train_set, test_set
    
    # de repente vale uma interseção aqui ja para extrair o que nao tem em algum set
    def get_items(self,train_set,test_set):
        train_items = train_set['item_desc'].drop_duplicates().tolist()
        test_items = train_set['item_desc'].drop_duplicates().tolist()
        return(train_items,test_items)
        
    def get_cluster_data(self,train_set,test_set):
    
        train_set_no_na=train_set.dropna(subset=['date_key','inputed_price_item30','inputed_price_item7','sales_avg_unit_price_with_tax'])
    
        df_clust = train_set.groupby('item_desc').agg(
        avg_distinct_days_per_month=('date_key', lambda x: x.dt.day.nunique() / x.dt.month.nunique()),
        gmv=('SALES_AMOUNT_WITH_TAX'.lower(), 'sum'),
        #unidades_totais_vendidas=('sales_qty_units_sold', 'sum'),
        ticket_medio=('SALES_AMOUNT_WITH_TAX'.lower(), lambda x: x.sum() / x.nunique())).reset_index() 
    
        numerical_variables = df_clust[['gmv','ticket_medio','avg_distinct_days_per_month']]
        kmeans_pipeline = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=6) 
        )
    
        kmeans_pipeline.fit(numerical_variables)
        df_clust['cluster_labels'] = kmeans_pipeline.predict(numerical_variables)
    
        cluster_centers = kmeans_pipeline.named_steps['kmeans'].cluster_centers_
        centers_df = pd.DataFrame(cluster_centers, columns=numerical_variables.columns)
        centers_df['cluster_labels'] = range(len(cluster_centers))
        cluster_labels_mapping = {label: letter for label, letter in zip(centers_df['cluster_labels'], string.ascii_uppercase)}
        centers_df['cluster'] = centers_df['cluster_labels'].map(cluster_labels_mapping)
        cluster_labels_mapping = {label: letter for label, letter in zip(centers_df['cluster_labels'], string.ascii_uppercase)}
        df_clust['cluster'] = df_clust['cluster_labels'].map(cluster_labels_mapping)
    
        df_clust.reset_index(inplace=True)
    
        df_clust = df_clust[['item_desc','cluster']] 
    
        train_set = pd.merge(train_set,df_clust,on='item_desc', how='left')
        train_set['cluster'] = train_set['cluster'].fillna('no_cluster')
        test_set = pd.merge(test_set,df_clust,on='item_desc', how='left')
        test_set['cluster'] = test_set['cluster'].fillna('no_cluster')
    
        return(train_set,test_set,centers_df)
    
    # incluir aqui gerenciamento de tipos
    def format_for_modeling(self,train_set,test_set):
        x_train = train_set.drop(['sales_qty_units_sold','sales_amount_with_tax'],axis=1) 
        x_test = test_set.drop(['sales_qty_units_sold','sales_amount_with_tax'],axis=1)

        y_train = train_set['sales_qty_units_sold']
        y_test = test_set['sales_qty_units_sold'] 
    
        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
    
    
    def process_data(self, data): # deixei feio porque tava quebrando e nao sabia oq era
        columns = ['date_key','item_desc', 'department_desc', 'group_desc', 'class_desc',
                          'sub_class_desc', 'brand_type_desc','permanent_price','promotion_price',
                          'distinct_dynamic_name_count','distinct_media_name_count','sales_avg_unit_price_with_tax',
                              'has_oferta_simples', 'has_desconto_grupo','sales_qty_units_sold',
                          'has_desconto_unidade', 'media_nacional', 'media_cupom', 'media_tv',
                          'media_lamina', 'media_youtube', 'media_celula', 'media_sinalv',
                          'media_sobrep', 'media_tabloide', 'has_leve_x_pague_y', 'has_depor',
                          'day_of_month', 'week_of_month', 'day_of_week', 'delta_days_promo','month_of_year',
                          'inputed_price_item7','inputed_price_item30', 'inputed_price_class7','inputed_sales7',
                   'inputed_sales30',
                   'inputed_price_class30','inputed_price_group7','inputed_price_group30','sales_amount_with_tax']
        data_filtered=data[columns]
        train_set, test_set = self.process_split_format(data_filtered)
        train_set, test_set = self.split_by_date(train_set, test_set)
        train_items, test_items = self.get_items(train_set,test_set)
        train_set, test_set,centers_df = self.get_cluster_data(train_set,test_set)
        formatted_data = self.format_for_modeling(train_set, test_set)
        return {
            'train_set': train_set, 
            'test_set': test_set,
            'train_set_items': train_items,
            'test_set_items': test_items,
            'cluster_centers': centers_df,
            'x_train': formatted_data['x_train'],
            'x_test': formatted_data['x_test'],
            'y_train': formatted_data['y_train'],
            'y_test': formatted_data['y_test']
        }

# aqui vale incluir vars de modelo e tunagem
class ML():
    def __init__(self, data):
        self.data = data
        
    def modeling_pipeline(self,data):
        # isso ta feio, parecendo que podia ser melhor
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        
        test_dates =x_test['date_key'].tolist()
        
        x_train.drop(columns=['date_key'],inplace=True)
        x_test.drop(columns=['date_key'],inplace=True)
    
        numerical_cols = x_train.select_dtypes(include=['number']).columns
        categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns

    
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
      
        lgbm = lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=100, learning_rate=0.1, random_state=42)
    
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', lgbm)])
    
        pipeline.fit(x_train, y_train)
    
        y_pred = pipeline.predict(x_test)
    
        y_pred_list = y_pred.tolist()
        y_test_list = y_test.tolist()
    
        output_df = x_test
        output_df['y_pred'] = y_pred_list
        output_df['sales_qty'] = y_test
        output_df['date_key'] = test_dates
        
        return(pipeline,output_df)

    def process_model(self, data):
            pipeline,output_df = self.modeling_pipeline(data)
            return {
                'model': pipeline,
                'df_predictions': output_df
                }
