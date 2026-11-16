"""SQLite storage for members, the approval audit trail and small key-values.
Swap for Postgres in production (same functions)."""
import json, os, sqlite3, time
from contextlib import closing


def _c():
    path = os.getenv("DB_PATH", "data/predictmed.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    return c


def init():
    with closing(_c()) as c:
        c.executescript("""
        create table if not exists members(id integer primary key autoincrement, name text unique, age int,
            sex text, uacr real, nsaid int, days_since_lab int, egfr text);
        create table if not exists audit(id integer primary key autoincrement, ts text, actor text,
            action text, member text, detail text);
        create table if not exists kv(k text primary key, v text);""")
        c.commit()


def add_member(m) -> bool:
    with closing(_c()) as c:
        cur = c.execute("insert or ignore into members(name,age,sex,uacr,nsaid,days_since_lab,egfr) values(?,?,?,?,?,?,?)",
                        (m["name"], m["age"], m.get("sex", "U"), m["uacr"], int(m["nsaid"]), m["days_since_lab"], json.dumps(m["egfr"])))
        c.commit()
        return cur.rowcount > 0


def _row(r):
    d = dict(r)
    d["egfr"] = json.loads(d["egfr"])
    return d


def members():
    with closing(_c()) as c:
        return [_row(r) for r in c.execute("select * from members order by id")]


def member(mid):
    with closing(_c()) as c:
        r = c.execute("select * from members where id=?", (mid,)).fetchone()
        return _row(r) if r else None


def audit(actor, action, member_name, detail=""):
    with closing(_c()) as c:
        c.execute("insert into audit(ts,actor,action,member,detail) values(?,?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), actor, action, member_name, detail))
        c.commit()


def audit_log(limit=50):
    with closing(_c()) as c:
        return [dict(r) for r in c.execute("select * from audit order by id desc limit ?", (limit,))]


def kv_get(k, default=None):
    with closing(_c()) as c:
        r = c.execute("select v from kv where k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default


def kv_set(k, v):
    with closing(_c()) as c:
        c.execute("insert or replace into kv(k,v) values(?,?)", (k, json.dumps(v)))
        c.commit()
