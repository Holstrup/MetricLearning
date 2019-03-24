import sqlite3
import numpy as np
import io


#Helper functions
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)



def reinitialize_table():
    """
    Creates Table in database.db file and drops it if it exists already
    """
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect('data.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS embeddings")
    cur.execute("CREATE TABLE embeddings(embedding ARRAY, label VARCHAR(10))")

def connect():
    """
    Connects to the database

    :return: Cursor, Connection
    """
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    con = sqlite3.connect('data.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    return cur, con


def add_encoding(classification, label="Null"):
    """
    Adds an encoding with label to the database. If not label is given it will be Null

    :param classification: Encoding array of dimension (128,)
    :param label: Label of the encoding given as a string.
    """
    cursor, connection = connect()
    if label == "Null":
        query = "INSERT INTO embeddings(embedding, label) VALUES(?, NULL)"
        cursor.execute(query, (classification,))
    else:
        query = "INSERT INTO embeddings(embedding, label) VALUES(?, ?)"
        cursor.execute(query, (classification, label,))
    connection.commit()
    connection.close()

