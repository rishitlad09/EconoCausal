from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO

app = FastAPI(
    title="EconoCausal API",
    description="Backend for campaign data and causal analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "EconoCausal",
        "status": "running",
        "message": "Backend API is ready",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/dataset/summary")
async def dataset_summary(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV contains no data rows.",
            )

        result = {
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "missing_values": int(df.isna().sum().sum()),
        }

        if "purchased" in df.columns:
            result["purchase_rate_percent"] = round(
                float(df["purchased"].mean() * 100), 2
            )

        if "treatment" in df.columns:
            result["customers_offered_discount"] = int(
                (df["treatment"] == 1).sum()
            )

        if "revenue" in df.columns:
            result["average_revenue"] = round(
                float(df["revenue"].mean()), 2
            )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the CSV: {exc}",
        ) from exc
