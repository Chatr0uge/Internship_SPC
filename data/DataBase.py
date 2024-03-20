import sqlite3
import numpy as np
import io
from typing import Iterable

class DataBase:
    
    def __init__(self, path : str, compressor = 'zlib') : 
        
        self.conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()
        self.c.execute("""CREATE TABLE IF NOT EXISTS data (id  INT PRIMARY KEY,
                        amplitude real,
                        lcx real,
                        lcy real,
                        bm real,
                        dt float,
                        pulse_list array,
                        delay_list array, 
                        pulse_width array,
                        skewness array,
                        kurtosis array,
                        mean_signal array,
                        variance array)""")
        
        self.conn.commit()
        self.compressor = compressor
        
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", lambda x: np.load(io.BytesIO(x), allow_pickle = True))
        
    def adapt_array(self,arr):
            """
            http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
            """
            # zlib uses similar disk size that Matlab v5 .mat files
            # bz2 compress 4 times zlib, but storing process is 20 times slower.
            out = io.BytesIO()
            np.save(out, arr)
            return sqlite3.Binary(out.getvalue())

    def convert_array(self,text):
        out = io.BytesIO(text)
        out.seek(0)
        out = io.BytesIO(out.read().decode(self.compressor))
        return np.load(out)
    
    def insert(self, data : dict) : 
        d = (data['id'], data['amplitude'], data['lcx'], data['lcy'], data['bm'], data['dt'], data['pulse_list'], data['delay_list'], data['pulse_width'], data['skewness'], data['kurtosis'], data['mean_signal'], data['variance'])
        self.c.execute("""INSERT INTO data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (data['id'], data['amplitude'], data['lcx'], data['lcy'], data['bm'], data['dt'], data['pulse_list'], data['delay_list'], data['pulse_width'], data['skewness'], data['kurtosis'], data['mean_signal'], data['variance']))
        self.conn.commit()
        
    def add_column(self, column_name : str, column : Iterable, type_column = 'real', replace = False) : 
        
        if column_name in np.array(self.c.execute(f'PRAGMA table_info(data)').fetchall())[:,1] and not replace : 
            
            raise ValueError('Column already exists... \n If you want to replace it, use replace = True')
                        
        else : 
            if not replace :
                self.c.execute(f'ALTER TABLE data ADD COLUMN {column_name} {type_column}')
                
            for i, row in enumerate(column) : 
                self.c.execute(f'UPDATE data SET {column_name} = ? WHERE id = ?', (row, i))
    
            self.conn.commit()
            
            
            
    def apply_function(self, function, column_name, new_column : str = None,  replace_original = False, replace_new = False, type_column = 'real') : 
        
        column = np.array(self.c.execute(f'SELECT {column_name} FROM data').fetchall())[:,0]
        column = np.apply_along_axis(function,1, column)
        if new_column is None : 
            new_column = column_name
            
            if replace_original : 
                self.add_column(new_column, column, replace = True, type_column=type_column)
        else :  
            
            if replace_original :
                raise ValueError('You cannot replace a column and create a new one at the same time')
            
            if  new_column in np.array(self.c.execute(f'PRAGMA table_info(data)').fetchall())[:,1]  and replace_new :
                self.add_column(new_column, column, replace = True, type_column =type_column)
                print(column.shape, type_column)
                
            if new_column in np.array(self.c.execute(f'PRAGMA table_info(data)').fetchall())[:,1] and not replace_new :
                raise ValueError('Column already exists... \n If you want to replace it, use replace_new = True')
                
            if  not (new_column in np.array(self.c.execute(f'PRAGMA table_info(data)').fetchall())[:,1]) and not replace_new : 
                self.add_column(new_column, column, replace = replace_original, type_column=type_column)
            
            if  not (new_column in np.array(self.c.execute(f'PRAGMA table_info(data)').fetchall())[:,1]) and replace_new : 
                raise ValueError('You cannot replace a column that does not exist')
        
        
    def to_pandas(self) : 
        
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM data", self.conn)