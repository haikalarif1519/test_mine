from sqlalchemy import create_engine


def get_engine(config: dict):
    db = config["database"]
    conn = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    return create_engine(conn, pool_pre_ping=True)
