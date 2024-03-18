import sqlite3
import numpy as np
import io

class DataBase:
    
    def __init__(self, path : str, compressor = 'zlib') : 
        
        self.conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()
        self.c.execute("""CREATE TABLE IF NOT EXISTS data (id  INT PRIMARY KEY,
                        amplitude real,
                        lcx real,
                        lcy real,
                        bm real,
                        dt real,
                        pulse_list array,
                        delay_list array, 
                        pulse_width array, 
                        skewness array,
                        kurtosis array,
                        mean_signal array,
                        variance array)
                        """)
        
        self.conn.commit()
        self.compressor = compressor
        
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", lambda x: np.load(io.BytesIO(x)))
        
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
        self.c.execute("""INSERT INTO data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (data['id'], data['amplitude'], data['lcx'], data['lcy'], data['bm'], data['dt'], data['pulse_list'],
                                                                           data['delay_list'], data['pulse_width'], data['skewness'], data['kurtosis'], data['mean_signal'], data['variance']))
        self.conn.commit()

    