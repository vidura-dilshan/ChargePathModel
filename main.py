from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import googlemaps
import polyline
import pickle
import io
import math
import os
import traceback

# ── 1. DQN Architecture (must match training exactly) ────────────────────────
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.net(x)

import __main__
setattr(__main__, "DQN", DQN)

# ── 2. CPU-Safe Pickle Loader ─────────────────────────────────────────────────
# This fixes: "Attempting to deserialize object on CUDA but cuda is not available"
class CPUUnpickler(pickle.Unpickler):
    """Remaps any CUDA storage to CPU during unpickling."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b),
                map_location=torch.device('cpu'),
                weights_only=False
            )
        return super().find_class(module, name)

# ── 3. App & Google Maps ──────────────────────────────────────────────────────
app = FastAPI()


GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

try:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    print("✅ Google Maps Client Initialized")
except Exception as e:
    print(f"❌ Google Maps Error: {e}")

# ── 4. Load & Preprocess Station Data ────────────────────────────────────────
FILE_ID = "17T4kgzEm5c22SotFISscbY7QvFSbzsHb"
URL     = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

EXPECTED_COLUMNS = [
    'station_id', 'station_name', 'latitude', 'longitude',
    'fast_charging', 'available_plugs', 'charging_power',
    'cost_per_kw', 'average_waiting_time', 'charging_duration',
    'status', 'connector_slots', 'supported_connector_types'
]

print("📦 Loading station data...")
try:
    df = pd.read_excel(URL)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if 'latitude' not in df.columns:
        df.columns = EXPECTED_COLUMNS[:len(df.columns)]

    for col in ['latitude', 'longitude', 'available_plugs', 'charging_power',
                'cost_per_kw', 'average_waiting_time', 'charging_duration', 'connector_slots']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['latitude', 'longitude'], inplace=True)

    df['charging_power']       = df['charging_power'].fillna(7.0)
    df['cost_per_kw']          = df['cost_per_kw'].fillna(df['cost_per_kw'].median())
    df['average_waiting_time'] = df['average_waiting_time'].fillna(10.0)
    df['charging_duration']    = df['charging_duration'].fillna(60.0)
    df['connector_slots']      = df['connector_slots'].fillna(1).astype(int)
    df['available_plugs']      = df['available_plugs'].fillna(1).astype(int)
    df['fast_charging']        = df['fast_charging'].astype(str).str.lower().isin(['yes', 'true', '1'])

    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower() == 'active'].reset_index(drop=True)

    df['supported_connector_types'] = df['supported_connector_types'].astype(str).apply(
        lambda x: [t.strip() for t in x.split(',')] if x.lower() != 'nan' else []
    )

    # Normalised columns required by quality scorer
    df['norm_power'] = df['charging_power']       / df['charging_power'].max()
    df['norm_cost']  = 1.0 - (df['cost_per_kw']  / df['cost_per_kw'].max())
    df['norm_wait']  = 1.0 - (df['average_waiting_time'] / df['average_waiting_time'].max())
    df['norm_plugs'] = df['available_plugs']      / df['available_plugs'].max()

    STATION_LOCATIONS = torch.tensor(
        df[['latitude', 'longitude']].values, dtype=torch.float32
    )
    print(f"✅ {len(df)} active stations loaded.")

except Exception as e:
    print(f"❌ Data load error: {e}")
    traceback.print_exc()
    df                = pd.DataFrame()
    STATION_LOCATIONS = torch.tensor([])

# ── 5. Load Pickle Model (CPU-safe) ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "AIModels", "ev_routing_agent.pkl")

policy_net         = None
load_error_message = ""

if os.path.exists(MODEL_PATH):
    print(f"📂 Found model: {MODEL_PATH}")
    try:
        # ✅ Use CPUUnpickler — fixes the CUDA→CPU deserialization error
        with open(MODEL_PATH, 'rb') as f:
            model_data = CPUUnpickler(f).load()

        cfg        = model_data["model_config"]
        policy_net = DQN(
            cfg["input_size"],
            cfg["hidden_size"],
            cfg["output_size"]
        )
        policy_net.load_state_dict(model_data["state_dict"])
        policy_net.eval()
        print("✅ Model loaded successfully on CPU!")
        print(f"   Input : {cfg['input_size']}  "
              f"Hidden: {cfg['hidden_size']}  "
              f"Output: {cfg['output_size']}")

    except Exception as e:
        load_error_message = str(e)
        print(f"❌ Model load error: {e}")
        traceback.print_exc()
else:
    load_error_message = f"File not found: {MODEL_PATH}"
    print(f"❌ {load_error_message}")

# ── 6. Helper Functions ───────────────────────────────────────────────────────
def vectorized_haversine(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    R    = 6371.0
    p1   = torch.deg2rad(pos1)
    p2   = torch.deg2rad(pos2)
    dlat = p2[:, 0] - p1[:, 0]
    dlon = p2[:, 1] - p1[:, 1]
    a    = (torch.sin(dlat / 2)**2
            + torch.cos(p1[:, 0]) * torch.cos(p2[:, 0])
            * torch.sin(dlon / 2)**2)
    return R * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def is_connector_compatible(station_idx: int, required_connector: str) -> bool:
    if not required_connector:
        return True
    supported = df.iloc[station_idx]['supported_connector_types']
    return any(required_connector.strip().lower() == s.lower() for s in supported)


def station_quality_score(station_idx: int) -> float:
    row = df.iloc[station_idx]
    return float(
        0.35 * row['norm_power'] +
        0.25 * row['norm_cost']  +
        0.25 * row['norm_wait']  +
        0.15 * row['norm_plugs']
    )


def compute_bearing(p1: torch.Tensor, p2: torch.Tensor) -> float:
    lat1, lon1 = torch.deg2rad(p1)
    lat2, lon2 = torch.deg2rad(p2)
    x = torch.sin(lon2 - lon1) * torch.cos(lat2)
    y = (torch.cos(lat1) * torch.sin(lat2)
         - torch.sin(lat1) * torch.cos(lat2) * torch.cos(lon2 - lon1))
    return float((torch.atan2(x, y) / math.pi + 1) / 2)


def get_route_and_filter_stations(
    start_point: str,
    end_point: str,
    buffer_km: float = 5.0,
    required_connector: str = None
):
    directions = gmaps.directions(
        origin=start_point,
        destination=end_point,
        mode="driving",
        units="metric",
        alternatives=False
    )
    if not directions:
        return None, [], {}

    leg        = directions[0]['legs'][0]
    route_meta = {
        'distance_text': leg['distance']['text'],
        'distance_km':   leg['distance']['value'] / 1000,
        'duration_text': leg['duration']['text'],
    }

    path_points  = polyline.decode(directions[0]['overview_polyline']['points'])
    path_tensor  = torch.tensor(path_points, dtype=torch.float32)
    station_locs = torch.tensor(
        df[['latitude', 'longitude']].values, dtype=torch.float32
    )

    valid_indices = []
    for i in range(len(df)):
        station_rep = station_locs[i].unsqueeze(0).expand(len(path_tensor), -1)
        min_dist    = vectorized_haversine(path_tensor, station_rep).min().item()
        if min_dist <= buffer_km and is_connector_compatible(i, required_connector):
            valid_indices.append(i)

    return path_points, valid_indices, route_meta

# ── 7. Request Schema ─────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    start_point:          str
    end_point:            str
    max_range_km:         float
    current_battery_pct:  float
    required_connector:   str = "Type 2"

# ── 8. /plan_route endpoint ───────────────────────────────────────────────────
@app.post("/plan_route")
def plan_route(request: RouteRequest):

    if policy_net is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded: {load_error_message}"
        )
    if df.empty:
        raise HTTPException(status_code=500, detail="Station data not loaded")

    start_point         = request.start_point
    end_point           = request.end_point
    max_range_km        = request.max_range_km
    current_battery_pct = request.current_battery_pct
    required_connector  = request.required_connector

    result_list = []

    try:
        route_points, valid_station_indices, route_meta = get_route_and_filter_stations(
            start_point, end_point,
            buffer_km=5.0,
            required_connector=required_connector
        )

        if not route_points:
            return {"Result": [], "Message": "No route found."}

        if not valid_station_indices:
            return {
                "Result": [],
                "Message": (
                    f"No '{required_connector}' stations found along this route. "
                    f"Try a wider buffer or different connector type."
                )
            }

        current_pos             = torch.tensor(route_points[0],  dtype=torch.float32)
        target_pos              = torch.tensor(route_points[-1], dtype=torch.float32)
        current_effective_range = max_range_km * (current_battery_pct / 100.0)

        visited   = set()
        steps     = 0
        max_steps = 10

        while steps < max_steps:

            dist_to_end = vectorized_haversine(
                current_pos.unsqueeze(0), target_pos.unsqueeze(0)
            ).item()

            if dist_to_end <= current_effective_range:
                break

            # 8-feature state vector
            batt_pct  = current_effective_range / max_range_km
            dist_norm = min(dist_to_end / 500.0, 1.0)
            bearing   = compute_bearing(current_pos, target_pos)

            curr_state = torch.tensor([
                current_pos[0], current_pos[1],
                target_pos[0],  target_pos[1],
                current_effective_range,
                dist_norm, bearing, batt_pct
            ], dtype=torch.float32)

            with torch.no_grad():
                q_values = policy_net(curr_state.unsqueeze(0))
                mask     = torch.full_like(q_values, -float('inf'))

                for idx in valid_station_indices:
                    if idx in visited:
                        continue

                    st_loc          = STATION_LOCATIONS[idx]
                    dist_to_station = vectorized_haversine(
                        current_pos.unsqueeze(0), st_loc.unsqueeze(0)
                    ).item()

                    if dist_to_station > current_effective_range:
                        continue

                    dist_station_to_target = vectorized_haversine(
                        st_loc.unsqueeze(0), target_pos.unsqueeze(0)
                    ).item()
                    if dist_station_to_target >= dist_to_end - 1.0:
                        continue

                    mask[0, idx] = station_quality_score(idx) * 2.0

                if torch.max(mask) == -float('inf'):
                    return {
                        "Result": result_list,
                        "Message": (
                            f"No reachable '{required_connector}' stations from current position. "
                            f"Range remaining: {current_effective_range:.1f} km"
                        )
                    }

                action = (q_values + mask).argmax().item()

            visited.add(action)
            steps += 1

            chosen_station = STATION_LOCATIONS[action]
            dist_hop       = vectorized_haversine(
                current_pos.unsqueeze(0), chosen_station.unsqueeze(0)
            ).item()

            # ✅ Output schema — identical to your original
            stop_info = {
                "Stop":                 steps,
                "StationName":          str(df.iloc[action]['station_name']),
                "DistancetoFind":       round(dist_hop, 2),
                "NeedChargePercentage": "100%",
                "Latitude":             float(chosen_station[0].item()),
                "Longitude":            float(chosen_station[1].item())
            }
            result_list.append(stop_info)

            current_pos             = chosen_station
            current_effective_range = max_range_km

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Runtime error: {e}")

    return {"Result": result_list}

# ── 9. Health check ───────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status":          "online",
        "model_loaded":    policy_net is not None,
        "stations_loaded": len(df),
    }