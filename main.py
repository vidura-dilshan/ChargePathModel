
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

# ── 1. DQN Architecture────────────────────────
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
# GOOGLE_MAPS_API_KEY = "AIzaSyALER_NJqGFdwseum4UGUk_wTTYZbGK-es"
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

# ── 7. Single-Leg Planner (used by both outbound & return) ────────────────────
def _plan_leg(
    start_pos: torch.Tensor,
    target_pos: torch.Tensor,
    effective_range: float,
    max_range_km: float,
    valid_station_indices: list,
    required_connector: str,
    max_steps: int = 10
) -> dict:
    """
    Plans a single leg (outbound OR return) using the DQN policy net.

    Battery tracking:
    - Each hop consumes range: current_effective_range -= dist_hop
    - Charging at a station resets range to max_range_km (100%)
    - When destination is reached, remaining_range_km = range - dist_to_end

    Returns
    -------
    dict with keys:
        stops               : list[dict]  – ordered charging stop dicts
        reached_destination : bool        – True if the destination is reachable
        remaining_range_km  : float       – range left upon arrival (0 if not reached)
    """
    current_pos             = start_pos.clone()
    current_effective_range = effective_range
    visited                 = set()
    steps                   = 0
    result_list             = []

    while steps < max_steps:

        dist_to_end = vectorized_haversine(
            current_pos.unsqueeze(0), target_pos.unsqueeze(0)
        ).item()

        # Can reach destination directly — done
        if dist_to_end <= current_effective_range:
            remaining = current_effective_range - dist_to_end
            return {
                "stops":               result_list,
                "reached_destination": True,
                "remaining_range_km":  round(remaining, 2),
            }

        # Build 8-feature state vector
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
                if dist_station_to_target >= dist_to_end - 1.0:   # must save ≥ 1 km
                    continue

                mask[0, idx] = station_quality_score(idx) * 2.0

            if torch.max(mask).item() == -float('inf'):
                return {
                    "stops":               result_list,
                    "reached_destination": False,
                    "remaining_range_km":  0.0,
                }

            action = (q_values + mask).argmax().item()

        visited.add(action)
        steps += 1

        chosen_station = STATION_LOCATIONS[action]
        dist_hop       = vectorized_haversine(
            current_pos.unsqueeze(0), chosen_station.unsqueeze(0)
        ).item()

        # Deduct range consumed traveling to this station
        current_effective_range -= dist_hop

        stop_info = {
            "Stop":                 steps,
            "StationName":          str(df.iloc[action]['station_name']),
            "DistancetoFind":       round(dist_hop, 2),
            "NeedChargePercentage": "100%",
            "Latitude":             float(chosen_station[0].item()),
            "Longitude":            float(chosen_station[1].item()),
        }
        result_list.append(stop_info)

        # After charging at the station, battery is full again
        current_pos             = chosen_station
        current_effective_range = max_range_km

    # Exhausted max_steps without reaching destination
    return {
        "stops":               result_list,
        "reached_destination": False,
        "remaining_range_km":  0.0,
    }

# ── 7b. Round-Trip Preemptive Stop Planner ────────────────────────────────────
def _plan_round_trip_with_preemptive_stop(
    start_pos: torch.Tensor,
    end_pos: torch.Tensor,
    effective_range: float,
    max_range_km: float,
    valid_station_indices: list,
    required_connector: str,
) -> tuple:
    """
    Called when the naive plan fails for a round trip: the outbound reaches
    the destination, but the leftover battery is too low to reach any station
    on the return.

    Strategy:
    1. Find the last compatible station before the destination (closest to
       end_pos) that is reachable from the start with the given effective range
       (possibly via earlier stops from _plan_leg).
    2. Force a charging stop there (outbound stop).
    3. After charging to 100%, drive from the station to the destination.
       Remaining battery = max_range_km - dist(station, destination).
    4. Plan the return leg with that remaining battery.

    Returns
    -------
    (outbound_result, return_result) – both in the same format as _plan_leg
    """
    # ── Step 1: Find the best "last stop before destination" ──────────────
    # We want the station closest to end_pos (so the user arrives with
    # maximum remaining battery) that is reachable from start_pos.
    candidates = []
    for idx in valid_station_indices:
        st_loc = STATION_LOCATIONS[idx]
        dist_st_to_end = vectorized_haversine(
            st_loc.unsqueeze(0), end_pos.unsqueeze(0)
        ).item()
        dist_start_to_st = vectorized_haversine(
            start_pos.unsqueeze(0), st_loc.unsqueeze(0)
        ).item()
        # Station must be reachable from start (directly or via earlier stops)
        # and must be closer to end_pos than start_pos is
        dist_start_to_end = vectorized_haversine(
            start_pos.unsqueeze(0), end_pos.unsqueeze(0)
        ).item()
        if dist_st_to_end < dist_start_to_end:
            candidates.append((idx, dist_st_to_end, dist_start_to_st))

    if not candidates:
        # No candidates at all — return empty (truly no feasible plan)
        empty = {"stops": [], "reached_destination": False, "remaining_range_km": 0.0}
        return empty, empty

    # Sort by distance to destination (ascending = closest to end first)
    candidates.sort(key=lambda x: x[1])

    # ── Step 2: Try each candidate (closest to destination first) ─────────
    for idx, dist_st_to_end, dist_start_to_st in candidates:
        # After charging at this station, remaining battery upon arrival
        # at the destination = max_range_km - dist_st_to_end
        arrival_range = max_range_km - dist_st_to_end
        if arrival_range <= 0:
            continue  # Station is further from destination than max range

        # Plan outbound: start → this station (may need intermediate stops)
        st_loc = STATION_LOCATIONS[idx]
        outbound_to_station = _plan_leg(
            start_pos              = start_pos,
            target_pos             = st_loc,
            effective_range        = effective_range,
            max_range_km           = max_range_km,
            valid_station_indices  = valid_station_indices,
            required_connector     = required_connector,
        )

        if not outbound_to_station["reached_destination"]:
            continue  # Can't even reach this station

        # ── Build outbound stops: all intermediate stops + the forced stop ──
        forced_stop_num = len(outbound_to_station["stops"]) + 1
        # Distance to the forced station from the last position
        if outbound_to_station["stops"]:
            # Last stop position
            last_stop = outbound_to_station["stops"][-1]
            last_pos = torch.tensor(
                [last_stop["Latitude"], last_stop["Longitude"]],
                dtype=torch.float32
            )
        else:
            last_pos = start_pos
        dist_to_forced = vectorized_haversine(
            last_pos.unsqueeze(0), st_loc.unsqueeze(0)
        ).item()

        forced_stop = {
            "Stop":                 forced_stop_num,
            "StationName":          str(df.iloc[idx]['station_name']),
            "DistancetoFind":       round(dist_to_forced, 2),
            "NeedChargePercentage": "100%",
            "Latitude":             float(st_loc[0].item()),
            "Longitude":            float(st_loc[1].item()),
        }

        outbound_stops = outbound_to_station["stops"] + [forced_stop]
        outbound_result = {
            "stops":               outbound_stops,
            "reached_destination": True,
            "remaining_range_km":  round(arrival_range, 2),
        }

        # ── Plan return leg with the improved remaining battery ──────────
        return_leg = _plan_leg(
            start_pos              = end_pos,
            target_pos             = start_pos,
            effective_range        = arrival_range,
            max_range_km           = max_range_km,
            valid_station_indices  = valid_station_indices,
            required_connector     = required_connector,
        )

        if return_leg["reached_destination"]:
            return outbound_result, return_leg
        # If this candidate didn't work, try the next one (further from dest
        # but leaves more battery for the return)

    # ── Fallback: none of the candidates made the full round trip work ────
    # Return the best attempt (first candidate) anyway so the user sees
    # partial data rather than nothing.
    best_idx, best_dist_to_end, _ = candidates[0]
    arrival_range = max_range_km - best_dist_to_end
    st_loc = STATION_LOCATIONS[best_idx]

    outbound_to_station = _plan_leg(
        start_pos, st_loc, effective_range, max_range_km,
        valid_station_indices, required_connector,
    )
    forced_stop_num = len(outbound_to_station["stops"]) + 1
    if outbound_to_station["stops"]:
        last_stop = outbound_to_station["stops"][-1]
        last_pos = torch.tensor(
            [last_stop["Latitude"], last_stop["Longitude"]],
            dtype=torch.float32
        )
    else:
        last_pos = start_pos
    dist_to_forced = vectorized_haversine(
        last_pos.unsqueeze(0), st_loc.unsqueeze(0)
    ).item()

    forced_stop = {
        "Stop":                 forced_stop_num,
        "StationName":          str(df.iloc[best_idx]['station_name']),
        "DistancetoFind":       round(dist_to_forced, 2),
        "NeedChargePercentage": "100%",
        "Latitude":             float(st_loc[0].item()),
        "Longitude":            float(st_loc[1].item()),
    }
    outbound_result = {
        "stops":               outbound_to_station["stops"] + [forced_stop],
        "reached_destination": True,
        "remaining_range_km":  round(max(arrival_range, 0), 2),
    }
    return_leg = _plan_leg(
        end_pos, start_pos, max(arrival_range, 0), max_range_km,
        valid_station_indices, required_connector,
    )
    return outbound_result, return_leg


# ── 8. Request Schema ────────────────────────────────────────────────────────
class RouteRequest(BaseModel):
    start_point:          str
    end_point:            str
    max_range_km:         float
    current_battery_pct:  float
    required_connector:   str  = "Type 2"
    return_trip:          int  = 0          # 0 = one-way, 1 = round-trip

# ── 9. /plan_route endpoint ──────────────────────────────────────────────────
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
    return_trip         = request.return_trip          # 0 or 1

    try:
        route_points, valid_station_indices, route_meta = get_route_and_filter_stations(
            start_point, end_point,
            buffer_km=5.0,
            required_connector=required_connector
        )

        # ── Early-exit: no route ──────────────────────────────────────────
        if not route_points:
            result = {"outbound": {"stops": []}}
            if return_trip == 1:
                result["return"] = {"stops": []}
            return {"Result": result}

        # ── Early-exit: no compatible stations ────────────────────────────
        if not valid_station_indices:
            result = {"outbound": {"stops": []}}
            if return_trip == 1:
                result["return"] = {"stops": []}
            return {"Result": result}

        # ── Coordinates ───────────────────────────────────────────────────
        start_pos = torch.tensor(route_points[0],  dtype=torch.float32)
        end_pos   = torch.tensor(route_points[-1], dtype=torch.float32)
        effective_range = max_range_km * (current_battery_pct / 100.0)

        # ── Phase 1: Outbound (start → end) ──────────────────────────────
        outbound = _plan_leg(
            start_pos              = start_pos,
            target_pos             = end_pos,
            effective_range        = effective_range,
            max_range_km           = max_range_km,
            valid_station_indices  = valid_station_indices,
            required_connector     = required_connector,
        )

        # ── One-way trip (return_trip == 0) ───────────────────────────────
        if return_trip == 0:
            return {
                "Result": {
                    "outbound": {
                        "stops": outbound["stops"],
                    }
                }
            }

        # ── Phase 2: Return (end → start) ────────────────────────────────
        # Use the REMAINING battery from the outbound leg as the starting
        # range for the return trip.  The user does NOT charge at the
        # destination — they head back with whatever battery they have left.
        return_leg = {
            "stops": [], "reached_destination": False, "remaining_range_km": 0.0,
        }

        if outbound["reached_destination"]:
            remaining_range = outbound["remaining_range_km"]
            return_leg = _plan_leg(
                start_pos              = end_pos,
                target_pos             = start_pos,
                effective_range        = remaining_range,
                max_range_km           = max_range_km,
                valid_station_indices  = valid_station_indices,
                required_connector     = required_connector,
            )

        # ── Phase 2b: Return failed — force a preemptive outbound stop ───
        # If the return leg couldn't reach its destination AND the outbound
        # had no stops (i.e. user went straight through), the leftover
        # battery at the destination is too low to reach any station on the
        # way back.  Solution: insert a mandatory charging stop on the
        # outbound at the LAST reachable station BEFORE the destination.
        # After charging to 100% there, the user drives the short remaining
        # distance to the destination and arrives with much more battery,
        # making the return trip feasible.
        if not return_leg["reached_destination"]:
            outbound, return_leg = _plan_round_trip_with_preemptive_stop(
                start_pos, end_pos, effective_range, max_range_km,
                valid_station_indices, required_connector,
            )

        # ── Round-trip response ───────────────────────────────────────────
        return {
            "Result": {
                "outbound": {
                    "stops": outbound["stops"],
                },
                "return": {
                    "stops": return_leg["stops"],
                },
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Runtime error: {e}")

# ── 10. Health check ──────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status":          "online",
        "model_loaded":    policy_net is not None,
        "stations_loaded": len(df),
    }