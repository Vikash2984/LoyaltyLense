import io
import os
import warnings
from pathlib import Path
from typing import Literal

import httpx
import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
STATIC_DIR   = ROOT_DIR / "static"

_env = Path(__file__).resolve().parent / ".env"
load_dotenv(_env if _env.exists() else ROOT_DIR / ".env", override=False)

def _clean_url(raw: str) -> str:
    """Normalise the Supabase project URL: strip trailing slashes/suffixes,
    and auto-add https:// so a missing protocol never crashes httpx."""
    raw = raw.strip().rstrip("/")
    for sfx in ("/rest/v1", "/auth/v1"):
        if raw.endswith(sfx):
            raw = raw[: -len(sfx)]
    # FIX: auto-prepend https:// if the user omitted it in .env
    if raw and not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw

SUPABASE_URL      = _clean_url(os.getenv("SUPABASE_URL", ""))
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    warnings.warn(
        "\n⚠️  Supabase credentials missing! Create a .env file with:\n"
        "    SUPABASE_URL=https://<project>.supabase.co\n"
        "    SUPABASE_ANON_KEY=<anon-key>\n"
    )

ACCESS_COOKIE  = "ll_access_token"
REFRESH_COOKIE = "ll_refresh_token"
COOKIE_MAX_AGE = 60 * 60 * 24 * 7

REQUIRED_COLUMNS = [
    "age", "gender", "subscription_type", "watch_hours", "last_login_days",
    "device", "monthly_fee", "payment_method", "number_of_profiles",
    "avg_watch_time_per_day", "favorite_genre",
]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Loyalty Lens API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Pydantic models ───────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    age:                    int   = Field(..., ge=0, le=100)
    gender:                 Literal["Male", "Female", "Other"]
    subscription_type:      Literal["Basic", "Standard", "Premium"]
    watch_hours:            float = Field(..., ge=0)
    last_login_days:        int   = Field(..., ge=0)
    device:                 Literal["Mobile", "TV", "Laptop", "Tablet", "Desktop"]
    monthly_fee:            float = Field(..., ge=0)
    payment_method:         Literal["Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"]
    number_of_profiles:     int   = Field(..., ge=0, le=5)
    avg_watch_time_per_day: float = Field(..., ge=0)
    favorite_genre:         Literal["Action", "Sci-Fi", "Drama", "Horror", "Romance", "Comedy", "Documentary"]

class AuthInput(BaseModel):
    email: str
    password: str

# ── ML artifacts (loaded once at startup) ────────────────────────────────────
model        = joblib.load(ARTIFACT_DIR / "voting_classifier_model.joblib")
encoders     = joblib.load(ARTIFACT_DIR / "encoders.joblib")
imputers     = joblib.load(ARTIFACT_DIR / "imputers.joblib")
feature_cols = [c for c in joblib.load(ARTIFACT_DIR / "feature_columns.joblib") if c != "churned"]
background   = pd.read_csv(ARTIFACT_DIR / "shap_background.csv")
encoders.pop("churned", None)
imputers.pop("churned", None)

gtb_model   = model.named_estimators_["gtb"]
explainer   = shap.Explainer(gtb_model, background, model_output="probability")
_churn_idx  = list(model.classes_).index(0)   # cached — avoids recomputing per request
_retain_idx = list(model.classes_).index(1)

# ── ML helpers ────────────────────────────────────────────────────────────────
def readable(s: str) -> str:
    return s.replace("_", " ").title()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, imp in imputers.items():
        if col in df.columns:
            df[col] = imp.transform(df[[col]]).ravel()
    for col, enc in encoders.items():
        if col in df.columns:
            enc_df = pd.DataFrame(enc.transform(df[[col]]),
                                  columns=enc.get_feature_names_out([col]), index=df.index)
            df = pd.concat([df.drop(columns=[col]), enc_df], axis=1)
    return df.reindex(columns=feature_cols, fill_value=0)

def get_shap_df(x: pd.DataFrame) -> pd.DataFrame:
    vals = explainer(x).values
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    df = pd.DataFrame({"feature": x.columns.map(readable), "shap_value": vals[0]})
    df["abs"] = df["shap_value"].abs()
    df["impact"] = df["shap_value"].apply(
        lambda v: "Which Increases churn risk" if (v < 0 if _churn_idx == 0 else v > 0)
        else "Which Reduces churn risk"
    )
    return df.sort_values("abs", ascending=False)

def build_drivers(shap_df: pd.DataFrame, status: str) -> list[dict]:
    key  = "Increases" if status == "HIGH CHURN RISK" else "Reduces"
    rows = shap_df[shap_df["impact"].str.contains(key)].head(3)
    rows = rows if not rows.empty else shap_df.head(3)
    return [{"feature": r.feature, "impact": r.impact,
             "absolute_magnitude": float(r.abs), "raw_weight_value": float(r.shap_value)}
            for r in rows.itertuples()]

def row_summary(cols: list, vals: np.ndarray, status: str, conf: float) -> str:
    tops = [readable(cols[i]) for i in np.argsort(np.abs(vals))[-3:][::-1]]
    return f"{status} with {conf:.2%} confidence. Main factors: {', '.join(tops)}."

def run_predict(df: pd.DataFrame):
    x = preprocess(df)
    return x, model.predict(x).astype(int), model.predict_proba(x)

# ── Auth helpers ──────────────────────────────────────────────────────────────
def _check_config():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(500, "Supabase not configured — check your .env file.")

def _set_cookies(resp: Response, session: dict):
    kw = dict(max_age=COOKIE_MAX_AGE, httponly=True, secure=False, samesite="lax")
    resp.set_cookie(ACCESS_COOKIE, session["access_token"], **kw)
    if rt := session.get("refresh_token"):
        resp.set_cookie(REFRESH_COOKIE, rt, **kw)

def _clear_cookies(resp: Response):
    resp.delete_cookie(ACCESS_COOKIE)
    resp.delete_cookie(REFRESH_COOKIE)

async def _sb(method: str, path: str, *, token: str | None = None, body: dict | None = None) -> dict:
    _check_config()
    hdrs = {"apikey": SUPABASE_ANON_KEY}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.request(method, f"{SUPABASE_URL}/auth/v1{path}", headers=hdrs, json=body)
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot reach Supabase. Check that SUPABASE_URL in your .env is correct.")
    except httpx.TimeoutException:
        raise HTTPException(504, "Supabase request timed out. Try again.")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Network error contacting Supabase: {e}")
    if r.status_code >= 400:
        try:   detail = r.json().get("msg") or r.json().get("error_description") or r.json()
        except ValueError: detail = r.text
        raise HTTPException(r.status_code, detail)
    return r.json()

async def current_user(req: Request):
    if tok := req.cookies.get(ACCESS_COOKIE):
        try:    return await _sb("GET", "/user", token=tok), None
        except Exception: pass          # FIX: catches httpx errors, not just HTTPException
    if not (rt := req.cookies.get(REFRESH_COOKIE)):
        return None, None
    try:
        s = await _sb("POST", "/token?grant_type=refresh_token", body={"refresh_token": rt})
        return s.get("user"), s
    except Exception:
        return None, None

async def require_user(req: Request):
    user, session = await current_user(req)
    if not user:
        raise HTTPException(401, "Please sign in to continue.")
    return user, session

# ── Page helpers ──────────────────────────────────────────────────────────────
def _authed_redirect(user, session):
    """Return a /dashboard redirect (with refreshed cookies) if the user is logged in."""
    if not user:
        return None
    r = RedirectResponse("/dashboard", status_code=303)
    if session:
        _set_cookies(r, session)
    return r

# ── Page routes ───────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root(req: Request):
    user, session = await current_user(req)
    return _authed_redirect(user, session) or FileResponse(STATIC_DIR / "index.html")

@app.get("/home", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/signin", include_in_schema=False)
async def signin_page(req: Request):
    user, session = await current_user(req)
    return _authed_redirect(user, session) or FileResponse(STATIC_DIR / "signin.html")

@app.get("/signup", include_in_schema=False)
async def signup_page(req: Request):
    user, session = await current_user(req)
    return _authed_redirect(user, session) or FileResponse(STATIC_DIR / "signup.html")

@app.get("/dashboard", include_in_schema=False)
async def dashboard_page(req: Request):
    user, session = await current_user(req)
    if not user:
        return RedirectResponse("/signin", status_code=303)
    r = FileResponse(STATIC_DIR / "dashboard.html")
    if session:
        _set_cookies(r, session)
    return r

# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata")
def metadata():
    return {"required_columns": REQUIRED_COLUMNS, "classes": [int(c) for c in model.classes_]}

@app.get("/auth/me")
async def auth_me(req: Request):
    user, session = await current_user(req)
    if not user:
        return JSONResponse({"user": None})
    r = JSONResponse({"user": {"id": user.get("id"), "email": user.get("email")}})
    if session:
        _set_cookies(r, session)
    return r

@app.post("/auth/signup")
async def auth_signup(payload: AuthInput):
    s = await _sb("POST", "/signup", body=payload.model_dump())
    r = JSONResponse({"user": s.get("user"), "confirmed": bool(s.get("access_token"))})
    if s.get("access_token"):
        _set_cookies(r, s)
    return r

@app.post("/auth/signin")
async def auth_signin(payload: AuthInput):
    s = await _sb("POST", "/token?grant_type=password", body=payload.model_dump())
    r = JSONResponse({"user": s.get("user")})
    _set_cookies(r, s)
    return r

@app.post("/auth/signout")
async def auth_signout(req: Request):
    if tok := req.cookies.get(ACCESS_COOKIE):
        try:   await _sb("POST", "/logout", token=tok)
        except Exception: pass
    r = JSONResponse({"ok": True})
    _clear_cookies(r)
    return r

@app.post("/predict")
async def predict_single(customer: CustomerInput, req: Request):
    await require_user(req)
    x, preds, proba = run_predict(pd.DataFrame([customer.model_dump()]))
    pred   = int(preds[0])
    status = "HIGH CHURN RISK" if pred == 0 else "LOW CHURN RISK"
    conf   = float(proba[0][_churn_idx if pred == 0 else _retain_idx])
    drvrs  = build_drivers(get_shap_df(x), status)
    return {
        "prediction":            pred,
        "status":                status,
        "confidence":            conf,
        "churn_probability":     float(proba[0][_churn_idx]),
        "retention_probability": float(proba[0][_retain_idx]),
        "top_drivers":           drvrs,
        "explanation": f"{status} with {conf:.2%} confidence. Main factors: {', '.join(d['feature'] for d in drvrs)}.",
    }

@app.post("/predict-bulk")
async def predict_bulk(req: Request, file: UploadFile = File(...)):
    await require_user(req)
    fname, content = (file.filename or "").lower(), await file.read()
    try:
        if   fname.endswith(".csv"):            df = pd.read_csv(io.BytesIO(content))
        elif fname.endswith((".xlsx", ".xls")): df = pd.read_excel(io.BytesIO(content))
        else: raise HTTPException(400, "Please upload a CSV or Excel file.")
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, f"Could not read file: {e}") from e

    if missing := [c for c in REQUIRED_COLUMNS if c not in df.columns]:
        raise HTTPException(422, f"Missing columns: {missing}")

    x, preds, proba = run_predict(df)
    sv = explainer(x).values
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    records = []
    for i, pred in enumerate(preds):
        status = "HIGH CHURN RISK" if pred == 0 else "LOW CHURN RISK"
        conf   = float(proba[i][_churn_idx if pred == 0 else _retain_idx])
        records.append({
            "row_number": i + 1,
            **df.iloc[i][REQUIRED_COLUMNS].replace({np.nan: None}).to_dict(),
            "prediction":        int(pred),
            "status":            status,
            "churn_probability": float(proba[i][_churn_idx]),
            "confidence":        conf,
            "explanation":       row_summary(list(x.columns), sv[i], status, conf),
        })

    churn_ct = int((preds == 0).sum())
    global_drivers = (
        pd.DataFrame({"feature": list(x.columns), "importance": np.abs(sv).mean(axis=0)})
        .assign(feature=lambda d: d["feature"].map(readable))
        .sort_values("importance", ascending=False)
        .head(3)
        .to_dict(orient="records")
    )
    return {
        "total_records":           len(df),
        "likely_churned_profiles": churn_ct,
        "churn_rate":              churn_ct / len(df),
        "global_top_drivers":      global_drivers,
        "records":                 records,
    }

import sys
print(">>> SUPABASE_URL =", repr(SUPABASE_URL), file=sys.stderr)
print(">>> SUPABASE_KEY =", repr(SUPABASE_ANON_KEY[:20] if SUPABASE_ANON_KEY else "EMPTY"), file=sys.stderr)
