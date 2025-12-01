from sqlalchemy import create_engine
import pandas as pd

def get_engine():
    user = "admin"
    password = "admin123"
    host = "localhost"
    db = "securities_db"

    return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

def load_financial_data():
    engine = get_engine()

    query = """
        SELECT 
            `tahun`,
            `kode_perusahaan`,
            `kuartal`,
            `NPM`,
            `revneg`,
            `netprofneg`,
            `ihsg`,
            `lq45`,
            `netprofit`,
            `revenue`
        FROM NPM;
    """

    df = pd.read_sql(query, engine)
    return df
