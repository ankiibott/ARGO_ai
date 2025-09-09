import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import base64
import xarray as xr
from io import BytesIO
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

app = FastAPI()

global_df = None  # store uploaded dataset globally


def nc_to_dataframe(nc_file) -> pd.DataFrame:
    
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    lats = ds["LATITUDE"].values
    lons = ds["LONGITUDE"].values
    times = pd.to_datetime(ds["JULD"].values)

    depths = ds["PRES"].values
    temps = ds["TEMP"].values
    salts = ds["PSAL"].values

    records = []


    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        t = times[i]

        for j in range(depths.shape[1]):
            d = depths[i, j]
            temp = temps[i, j]
            sal = salts[i, j]

            if pd.isna(d) or pd.isna(temp) or pd.isna(sal):
                continue

            records.append({
                "latitude": float(lat),
                "longitude": float(lon),
                "time": pd.to_datetime(t),
                "depth": float(d),
                "temperature": float(temp),
                "salinity": float(sal),
            })

    df = pd.DataFrame.from_records(records)
    return df


def nlp_to_sql(query: str, table_name: str = "argo_data") -> str:
    sql = f"SELECT * FROM {table_name} WHERE 1=1"


    if "temperature" in query.lower():
        sql = sql.replace("*", "temperature, latitude, longitude, depth, time")
    elif "salinity" in query.lower():
        sql = sql.replace("*", "salinity, latitude, longitude, depth, time")

    lat_match = re.findall(r"(\d+\.\d+)", query)
    if len(lat_match) >= 2:
        lat_min, lat_max = lat_match[:2]
        sql += f" AND latitude BETWEEN {lat_min} AND {lat_max}"

 
    if "march 2023" in query.lower():
        sql += " AND time LIKE '2023-03%'"

    return sql + ";"


def plot_profile_image(df: pd.DataFrame, variable: str):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(df[variable], df["depth"], marker="o", linestyle="-")
    ax.invert_yaxis()
    ax.set_xlabel(variable.capitalize())
    ax.set_ylabel("Depth (dbar)")
    ax.set_title(f"{variable.capitalize()} Profile")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def compare_profiles_image(df: pd.DataFrame, variable: str):
    fig = px.line(
        df,
        x=variable,
        y="depth",
        color="latitude",
        markers=True,
        title=f"Comparison of {variable.capitalize()} Profiles"
    )
    fig.update_yaxes(autorange="reversed")
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf


def map_html(df: pd.DataFrame, lat_col="latitude", lon_col="longitude"):
    if df.empty:
        return None

    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

    for _, row in df.iterrows():
        folium.Marker(
            [row[lat_col], row[lon_col]],
            popup=f"Lat: {row[lat_col]}, Lon: {row[lon_col]}"
        ).add_to(m)

    return m._repr_html_()


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global global_df
    try:
        file_path = file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())


        engines = ["netcdf4", "h5netcdf", "scipy"]
        ds = None
        for eng in engines:
            try:
                ds = xr.open_dataset(file_path, engine=eng)
                print(f"Opened with engine: {eng}")
                break
            except Exception as e:
                print(f"Failed with {eng}: {e}")

        if ds is None:
            raise ValueError("Could not open file with any xarray engine.")


        global_df = nc_to_dataframe(file_path)


        csv_file = file_path.replace(".nc", ".csv")
        global_df.to_csv(csv_file, index=False)

        return {
            "message": f"File processed and saved as {csv_file}",
            "rows": len(global_df),
            "columns": list(global_df.columns)
        }

    except Exception as e:
        return {"message": f"❌ Error processing file: {str(e)}"}


@app.get("/")
def root():
    return {"message": "ARGO Prototype API is running 🚀"}


@app.post("/chatbot-response")
async def chatbot_response(query: str = Form(...)):
    global global_df
    if global_df is None:
        return JSONResponse(content={"message": "⚠️ Please upload data first. Need more help?"})

    df = global_df.copy()
    query_lower = query.lower()

    sql = nlp_to_sql(query)
    print("🔍 SQL generated:", sql)


    if "latitude BETWEEN" in sql:
        match = re.search(r"latitude BETWEEN ([\d\.]+) AND ([\d\.]+)", sql)
        if match:
            lat_min, lat_max = map(float, match.groups())
            df = df[(df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)]

    if "time LIKE" in sql and "2023-03" in sql:
        df = df[df["time"].astype(str).str.contains("2023-03")]

    if "salinity" in sql.lower():
        variable = "salinity"
    elif "temperature" in sql.lower():
        variable = "temperature"
    else:
        variable = None

    if df.empty:
        return {"message": "⚠️ No data found for this query. Need more help?"}


    if "compare" in query_lower:
        if variable:
            buf = compare_profiles_image(df, variable)
            return StreamingResponse(buf, media_type="image/png")

    if any(word in query_lower for word in ["plot", "profile", "depth"]):
        if variable:
            buf = plot_profile_image(df, variable)
            return StreamingResponse(buf, media_type="image/png")
        else:
            return {"message": "⚠️ Variable not found for plotting. Need more help?"}

    if "map" in query_lower:
        map_content = map_html(df)
        return HTMLResponse(content=map_content)


    return {
        "message": f"✅ Found {len(df)} records. Need more help?",
        "preview": df.head(5).to_dict(orient="records"),
        "sql": sql
    }
