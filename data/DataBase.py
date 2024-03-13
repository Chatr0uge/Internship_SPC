import sqlite3
import numpy as np
import io

class DataBase:
    
    def __init__(self, path : str, compressor = 'zlib') : 
        
        self.conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.c = self.conn.cursor()
        self.c.execute("""CREATE TABLE IF NOT EXISTS data (id integer,
                        amplitude real,
                        lc_x real,
                        lc_y real,
                        bm real,
                        time_list real,
                        pulse array, 
                        seed integer)""")
        
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
        
        self.c.execute("INSERT INTO data VALUES (?,?,?,?,?,?,?,?)", 
                       (data['id'],th data['amplitude'], data['lc_x'], data['lc_y'], data['bm'], data['time_list'], data['pulse'], data['seed']))
        self.conn.commit()
    